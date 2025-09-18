import streamlit as st
from dotenv import load_dotenv
import os, io, datetime as dt, shutil
from utils.transcribe import transcribe_audio
from utils.summarize import summarize_transcript
from utils.classify import decide_doc_type
from utils.export import render_markdown, save_markdown, markdown_to_pdf

# 추가: 오디오 처리
from pydub import AudioSegment
from pydub.utils import which
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
st.set_page_config(page_title="병원 회의용 음성 자동 노트", page_icon="🩺", layout="centered")
st.title("🩺 음성기반 자동 회의록/연구노트")
st.caption("업로드 → 전사 → 요약 → 서식 적용 → Markdown/PDF 저장")

# -------------------------------------------------
# 🔧 ffmpeg/ffprobe 보장 + 탐색 강화 + UI 리포트
# -------------------------------------------------
def _find_binary(name: str):
    # 1) 환경변수 최우선 (사용자가 직접 지정 가능)
    env_key = "FFMPEG_BINARY" if name == "ffmpeg" else ("FFPROBE_BINARY" if name == "ffprobe" else None)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)

    # 2) PATH에서 탐색
    path = which(name) or shutil.which(name)
    if path:
        return path

    # 3) 자주 쓰는 시스템 경로 후보들
    candidates = [
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/bin/{name}",
        f"/opt/homebrew/bin/{name}",  # (macOS 대비, Cloud에선 무의미하지만 안전)
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    return None

def ensure_ffmpeg() -> tuple[str, str]:
    """
    pydub가 사용할 ffmpeg/ffprobe 경로를 확실히 세팅하고 UI에 경로 출력.
    못 찾으면 Streamlit에서 에러로 중단.
    """
    ffmpeg_path  = _find_binary("ffmpeg")
    ffprobe_path = _find_binary("ffprobe")

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path

    if not ffmpeg_path or not ffprobe_path:
        msg = (
            "ffmpeg/ffprobe 실행파일을 찾지 못했습니다.\n\n"
            "• Streamlit Cloud라면 프로젝트 루트에 **packages.txt** 파일을 만들고 아래 두 줄을 넣은 뒤 재배포하세요:\n"
            "```\nffmpeg\nwkhtmltopdf\n```\n"
            "• 재배포 후에도 문제면, Cloud 셸에서 `/usr/bin/ffprobe -version` 또는 `/usr/local/bin/ffprobe -version`이 출력되는지 확인하세요.\n"
            "• (선택) 경로를 직접 지정하려면 환경변수 **FFMPEG_BINARY**, **FFPROBE_BINARY** 를 설정하세요."
        )
        st.error(msg)
        st.stop()

    st.info(f"🔎 ffmpeg: `{ffmpeg_path}`\n\n🔎 ffprobe: `{ffprobe_path}`")
    return ffmpeg_path, ffprobe_path

def load_audio_from_bytes(bytes_data: bytes, filename: str | None):
    """
    BytesIO로 읽을 때 포맷을 명시해야 안정적.
    ffprobe가 없어도 여기까지 오기 전에 ensure_ffmpeg에서 막힘.
    """
    ext = None
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    try:
        return AudioSegment.from_file(io.BytesIO(bytes_data), format=ext)
    except FileNotFoundError as e:
        # pydub 내부에서 ffprobe 호출 실패 시 FileNotFoundError로 전파됨
        raise FileNotFoundError(
            "오디오 로딩 중 ffprobe를 실행하지 못했습니다. 상단의 ffmpeg/ffprobe 경로 안내를 확인하세요."
        ) from e

# ------------------------------
# 고속 전사(병렬 청크) 헬퍼 함수
# ------------------------------
def fast_transcribe_ko_with_progress(bytes_data: bytes, filename: str, api_key: str | None,
                                     chunk_ms: int = 60_000, max_workers: int = 6) -> str:
    # 🔧 ffmpeg/ffprobe 보장 + 경로 리포트
    ensure_ffmpeg()

    # 1) 로드 & 다운샘플
    audio = load_audio_from_bytes(bytes_data, filename)
    audio = audio.set_frame_rate(16_000).set_channels(1)

    # 2) 청크 분할
    total_ms = len(audio)
    chunks = []
    for i, start in enumerate(range(0, total_ms, chunk_ms)):
        seg = audio[start: start + chunk_ms]
        chunks.append((i, seg))

    total = len(chunks)
    if total == 0:
        return ""

    # 3) 진행률 UI
    progress = st.progress(0)
    status = st.empty()
    done = 0

    # 4) per-chunk 전사 함수
    def transcribe_one(idx_seg):
        idx, seg = idx_seg
        buf = io.BytesIO()
        seg.export(buf, format="mp3", bitrate="32k")  # 업로드 시간 단축
        data = buf.getvalue()
        out = transcribe_audio(data, f"{idx}_{filename}.mp3", api_key=(api_key or None), language_hint="ko")
        text = out if isinstance(out, str) else (out.get("text", "") if isinstance(out, dict) else str(out))
        return idx, text

    # 5) 병렬 실행
    texts = [None] * total
    workers = min(max_workers, total)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(transcribe_one, ch) for ch in chunks]
        for fut in as_completed(futures):
            idx, text = fut.result()
            texts[idx] = text
            done += 1
            pct = int(done / total * 100)
            progress.progress(pct)
            status.write(f"전사 진행률: {done}/{total} 청크 완료 ({pct}%)")

    progress.empty()
    status.empty()
    return " ".join(t for t in texts if t)

# ------------------------------
# UI
# ------------------------------
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key (미입력 시 .env 사용)", type="password", value="")
    auto_detect = st.toggle("회의 타입 자동감지(일반/연구)", value=True)
    st.markdown("---")
    st.markdown("**PDF 엔진 메모**  \n`wkhtmltopdf`가 있으면 고품질 PDF, 없으면 단순 PDF로 저장합니다.")

st.subheader("1) 오디오 업로드")
uploaded = st.file_uploader(
    "오디오 파일 (mp3, m4a, wav, ogg, webm 등)",
    type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"]
)

st.subheader("2) 메타데이터")
col1, col2 = st.columns(2)
with col1:
    mt_title = st.text_input("제목", "의생명연구원 정기회의")
    mt_dt = st.text_input("일시", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    mt_place = st.text_input("장소", "의생명연구원 2층 회의실")
with col2:
    mt_att = st.text_input("참석자(역할)", "홍길동(PI), 김철수(기획), 박영희(연구)…")
    host = st.text_input("진행/서기", "진행: 홍길동 / 서기: 김철수")

meta = {
    "title": mt_title, "dt": mt_dt, "place": mt_place, "attendees": mt_att,
    "host": (host.split("/")[0].replace("진행:", "").strip() if host else ""),
    "scribe": (host.split("/")[-1].replace("서기:", "").strip() if host else ""),
    "project": ""
}

st.divider()

if uploaded is not None:
    # 업로드 직후 바이너리 경로 확인 및 보고
    ensure_ffmpeg()

    bytes_data = uploaded.getvalue()

    # ffprobe 없이 pydub 길이 계산 (내부 디코드 길이 사용)
    try:
        _aud = load_audio_from_bytes(bytes_data, uploaded.name)
        duration_sec = len(_aud) / 1000.0
        st.write(f"오디오 길이: {duration_sec/60:.1f}분")
        if duration_sec > 2 * 3600:
            st.error("데모 제한: 2시간을 초과한 파일은 처리하지 않습니다.")
            st.stop()
    except Exception as e:
        st.warning(f"길이 확인 실패(계속 진행 가능): {e}")

    if st.button("전사 → 요약 → 서식 적용 실행", type="primary"):
        with st.spinner("전사 중…"):
            transcript = fast_transcribe_ko_with_progress(
                bytes_data=bytes_data,
                filename=uploaded.name,
                api_key=(api_key or None),
                chunk_ms=60_000,
                max_workers=6
            )

        with st.spinner("요약/분류 중…"):
            summary = summarize_transcript(transcript, api_key=(api_key or None))
            doc_type = decide_doc_type(summary)

        if not auto_detect:
            doc_type = st.radio("문서 유형 선택", options=["general","research"], index=0, horizontal=True)

        st.success(f"문서 유형: {'연구노트' if doc_type=='research' else '일반 회의록'}")

        if doc_type == "research":
            enrich = summary.get("research_enrich", {}) if isinstance(summary, dict) else {}
            context = {
                "meta": meta,
                "summary": {
                    "brief": (summary.get("brief","") if isinstance(summary, dict) else ""),
                    "bullets": (summary.get("bullets",[]) if isinstance(summary, dict) else []),
                    "decisions": (summary.get("decisions",[]) if isinstance(summary, dict) else []),
                    "actions": (summary.get("actions",[]) if isinstance(summary, dict) else []),
                    "objective": enrich.get("objective",""),
                    "methods": enrich.get("methods",[]),
                    "results": enrich.get("results",[]),
                    "limitations": enrich.get("limitations",""),
                },
                "transcript": transcript
            }
            template_name = "research.md.j2"
            out_base = "연구노트_" + dt.datetime.now().strftime("%Y%m%d_%H%M")
        else:
            context = {
                "meta": meta,
                "summary": {
                    "brief": (summary.get("brief","") if isinstance(summary, dict) else ""),
                    "bullets": (summary.get("bullets",[]) if isinstance(summary, dict) else []),
                    "decisions": (summary.get("decisions",[]) if isinstance(summary, dict) else []),
                    "actions": (summary.get("actions",[]) if isinstance(summary, dict) else []),
                },
                "transcript": transcript
            }
            template_name = "meeting.md.j2"
            out_base = "회의록_" + dt.datetime.now().strftime("%Y%m%d_%H%M")

        md_text = render_markdown("templates", template_name, context)
        md_path = f"{out_base}.md"
        pdf_path = f"{out_base}.pdf"

        save_markdown(md_text, md_path)
        try:
            markdown_to_pdf(md_text, pdf_path)
            st.success("Markdown/PDF 저장 완료")
            with open(md_path, "rb") as f:
                st.download_button("📥 Markdown 다운로드", data=f, file_name=md_path, mime="text/markdown")
            with open(pdf_path, "rb") as f:
                st.download_button("📥 PDF 다운로드", data=f, file_name=pdf_path, mime="application/pdf")
        except Exception as e:
            st.error(f"PDF 변환 중 오류: {e}")
            with open(md_path, "rb") as f:
                st.download_button("📥 Markdown만 다운로드", data=f, file_name=md_path, mime="text/markdown")

        st.markdown("미리보기(요약)")
        st.code(md_text[:1500] + ("..." if len(md_text)>1500 else ""), language="markdown")
else:
    st.info("좌측에서 오디오를 업로드하면 시작할 수 있어요.")
