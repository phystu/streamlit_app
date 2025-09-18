import streamlit as st
from dotenv import load_dotenv
import os, io, datetime as dt, shutil, re, subprocess, json, tempfile
from utils.transcribe import transcribe_audio
from utils.summarize import summarize_transcript
from utils.classify import decide_doc_type
from utils.export import render_markdown, save_markdown, markdown_to_pdf

# 추가: 오디오 처리
from pydub import AudioSegment
from pydub.utils import which
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub.silence import detect_nonsilent


load_dotenv()
st.set_page_config(page_title="병원 회의용 음성 자동 노트", page_icon="🩺", layout="centered")
st.title("🩺 음성기반 자동 회의록/연구노트")
st.caption("업로드 → 전사 → 요약 → 서식 적용 → Markdown/PDF 저장")

# -------------------------------------------------
# 🔧 ffmpeg/ffprobe 보장 + 탐색 강화 + UI 리포트
# -------------------------------------------------
def _find_binary(name: str):
    env_key = "FFMPEG_BINARY" if name == "ffmpeg" else ("FFPROBE_BINARY" if name == "ffprobe" else None)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)
    path = which(name) or shutil.which(name)
    if path:
        return path
    candidates = [f"/usr/bin/{name}", f"/usr/local/bin/{name}", f"/bin/{name}", f"/opt/homebrew/bin/{name}"]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None

def ensure_ffmpeg() -> tuple[str, str]:
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
            "• (선택) 환경변수 **FFMPEG_BINARY**, **FFPROBE_BINARY** 로 절대경로 지정 가능."
        )
        st.error(msg); st.stop()
    st.info(f"🔎 ffmpeg: `{ffmpeg_path}`\n\n🔎 ffprobe: `{ffprobe_path}`")
    return ffmpeg_path, ffprobe_path

def load_audio_from_bytes(bytes_data: bytes, filename: str | None):
    ext = None
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    try:
        return AudioSegment.from_file(io.BytesIO(bytes_data), format=ext)
    except FileNotFoundError as e:
        raise FileNotFoundError("오디오 로딩 중 ffprobe 실행 실패. ffmpeg/ffprobe 경로를 확인하세요.") from e

# ------------------------------
# 길이 측정: ffprobe 우선, pydub 폴백
# ------------------------------
def _probe_duration_seconds_ffprobe(bytes_data: bytes, filename: str) -> float | None:
    ffprobe_path = getattr(AudioSegment, "ffprobe", None)
    if not ffprobe_path:
        return None
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(bytes_data); tmp_path = tmp.name
    try:
        cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_format", tmp_path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(out.decode("utf-8", errors="ignore"))
        dur = info.get("format", {}).get("duration")
        if dur is None: return None
        val = float(dur)
        return val if val > 0 else None
    except Exception:
        return None
    finally:
        try: os.remove(tmp_path)
        except: pass

def safe_get_duration_seconds(bytes_data: bytes, filename: str) -> float | None:
    val = _probe_duration_seconds_ffprobe(bytes_data, filename)
    if val and val > 0:
        return val
    try:
        aud = load_audio_from_bytes(bytes_data, filename)
        sec = len(aud) / 1000.0
        return sec if sec > 0 else None
    except Exception:
        return None

# ------------------------------
# 고속 전사(병렬 청크) 헬퍼 함수
# ------------------------------
def fast_transcribe_ko_with_progress(bytes_data: bytes, filename: str, api_key: str | None,
                                     chunk_ms: int = 60_000, max_workers: int = 6) -> str:
    ensure_ffmpeg()
    audio = load_audio_from_bytes(bytes_data, filename).set_frame_rate(16_000).set_channels(1)
    total_ms = len(audio)
    chunks = [(i, audio[start:start+chunk_ms]) for i, start in enumerate(range(0, total_ms, chunk_ms))]
    total = len(chunks)
    if total == 0:
        return ""
    progress = st.progress(0); status = st.empty()
    done = 0; failed = 0

    def transcribe_one(idx_seg):
        idx, seg = idx_seg
        try:
            buf = io.BytesIO()
            seg.export(buf, format="mp3", bitrate="32k")
            data = buf.getvalue()
            out = transcribe_audio(data, f"{idx}_{filename}.mp3", api_key=(api_key or None), language_hint="ko")
            text = out if isinstance(out, str) else (out.get("text", "") if isinstance(out, dict) else str(out))
            return idx, (text or "").strip()
        except Exception:
            return idx, ""

    texts = [None] * total
    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as ex:
        futures = [ex.submit(transcribe_one, ch) for ch in chunks]
        for fut in as_completed(futures):
            idx, text = fut.result()
            texts[idx] = text
            if not text: failed += 1
            done += 1
            pct = int(done / total * 100)
            progress.progress(pct)
            status.write(f"전사 진행률: {done}/{total} 청크 완료 ({pct}%)")
    progress.empty(); status.empty()
    if failed:
        st.warning(f"일부 청크 전사 실패: {failed}/{total}")
    full_text = "\n".join(t for t in texts if t)
    if len(full_text.strip()) < 30:
        raise RuntimeError("전사 결과가 비정상적으로 짧습니다. 파일/코덱/네트워크/키를 확인하세요.")
    return full_text

# ------------------------------
# 프로브 전사(앞 10초) + 유사도 검증
# ------------------------------
def probe_transcribe_10s(bytes_data: bytes, filename: str, api_key: str | None) -> str:
    """
    앞부분 '프로브 전사':
    - 첫 비무음 구간부터 최소 3초 이상, 기본 10초
    - 짧다는 에러가 오면 2배씩 최대 30초까지 확대 재시도
    - 16kHz/mono/16bit PCM(WAV)로 안정화
    """
    aud = load_audio_from_bytes(bytes_data, filename).set_frame_rate(16_000).set_channels(1).set_sample_width(2)
    total_ms = len(aud)
    if total_ms < 150:  # 0.15s 이하면 무의미
        return ""

    # 1) 첫 비무음 구간 찾기 (무음 시작 문제 회피)
    try:
        # 너무 타이트하지 않게: 평균보다 14dB 낮춤, 최소 -50dBFS
        thresh = max(-50, (aud.dBFS if aud.dBFS != float("-inf") else -60) - 14)
        regions = detect_nonsilent(aud, min_silence_len=150, silence_thresh=thresh)
    except Exception:
        regions = []

    start = max(0, regions[0][0] - 250) if regions else 0  # 약간 앞을 포함
    want_ms = 10_000
    want_ms = min(total_ms - start, want_ms)
    want_ms = max(want_ms, 3_000)  # 최소 3초

    def _export_wav(s: int, e: int) -> bytes:
        seg = aud[s:e]
        buf = io.BytesIO()
        # pcm_s16le 보장
        seg.export(buf, format="wav", parameters=["-acodec", "pcm_s16le"])
        return buf.getvalue()

    s, e = start, min(total_ms, start + want_ms)
    wav_bytes = _export_wav(s, e)

    # 2) 전사 시도 + 'too short' 자동 확대 재시도 (최대 30초)
    attempts = 0
    while True:
        try:
            txt = transcribe_audio(wav_bytes, "probe.wav", api_key=api_key, language_hint="ko")
            return (txt or "").strip()
        except Exception as err:
            msg = str(err)
            too_short = ("audio_too_short" in msg) or ("too short" in msg) or ("'seconds': 0" in msg)
            if too_short and (e - s) < 30_000:  # 30초까지 확대
                new_len = max((e - s) * 2, 15_000)  # 최소 15초로 넓힘
                new_len = min(new_len, 30_000, total_ms - s)
                if new_len <= (e - s):  # 더 못 넓히면 중단
                    return ""
                e = s + new_len
                wav_bytes = _export_wav(s, e)
                attempts += 1
                continue
            # 비무음이 전혀 없거나 다른 에러면 상위에서 경고 처리
            return ""


def _tokens(s: str) -> set:
    return set(re.findall(r"[가-힣A-Za-z0-9]{2,}", s))

def similar_enough(short_txt: str, long_txt: str, min_overlap_ratio: float = 0.15) -> bool:
    A, B = _tokens(short_txt.lower()), _tokens(long_txt.lower())
    if not A or not B: return False
    overlap = len(A & B) / max(len(A), 1)
    return overlap >= min_overlap_ratio

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

# 세션 상태: 업로드 변경 시 초기화
if 'last_upload_name' not in st.session_state:
    st.session_state.last_upload_name = None
if uploaded is not None and uploaded.name != st.session_state.last_upload_name:
    st.session_state.last_upload_name = uploaded.name
    for k in ("transcript", "summary", "doc_type"):
        st.session_state.pop(k, None)

if uploaded is not None:
    ensure_ffmpeg()
    ext = uploaded.name.split(".")[-1].lower()
    bytes_data = uploaded.getvalue()

    if not bytes_data or len(bytes_data) < 1024:
        st.error("업로드한 파일이 비어 있거나 너무 작습니다. 파일을 다시 업로드해 주세요.")
        st.stop()

    # 미리듣기(진짜 파일인지 육안 확인)
    st.audio(bytes_data, format=f"audio/{ext}")

    # 길이 표시(정확)
    duration_sec = None
    try:
        duration_sec = safe_get_duration_seconds(bytes_data, uploaded.name)
    except Exception as e:
        st.warning(f"길이 확인 중 예외: {e}")
    if duration_sec is None or duration_sec <= 0:
        st.warning("오디오 길이 확인 실패(계속 진행 가능). 파일 코덱/컨테이너 문제일 수 있어요.")
    else:
        st.write(f"오디오 길이: {duration_sec/60:.1f}분")
        if duration_sec > 2 * 3600:
            st.error("데모 제한: 2시간을 초과한 파일은 처리하지 않습니다.")
            st.stop()

    # 프로브 전사(앞 10초)
    probe_text = ""
    try:
        probe_text = probe_transcribe_10s(bytes_data, uploaded.name, api_key or None)
        st.markdown("**전사 프로브(앞 10초) 미리보기**")
        st.code(probe_text[:300] + ("..." if len(probe_text) > 300 else ""))
        if len(probe_text) < 5:
            st.warning("앞 10초 전사가 거의 비어 있습니다. 파일의 실제 내용/코덱을 확인하세요.")
    except Exception as e:
        st.warning(f"프로브 전사 실패(계속 가능): {e}")

    # 실행
    if st.button("전사 → 요약 → 서식 적용 실행", type="primary"):
        with st.spinner("전사 중…"):
            try:
                transcript = fast_transcribe_ko_with_progress(
                    bytes_data=bytes_data,
                    filename=uploaded.name,
                    api_key=(api_key or None),
                    chunk_ms=60_000,
                    max_workers=6
                )
            except Exception as e:
                st.error(f"전사 실패: {e}")
                st.stop()

        # 품질 가드
        if not transcript or len(transcript.strip()) < 30:
            st.error("전사 결과가 비어 있거나 너무 짧습니다. 파일/코덱/네트워크 문제를 확인하세요.")
            st.stop()
        try:
            if probe_text and not similar_enough(probe_text, transcript):
                st.error("전사 결과가 오디오(앞 10초) 내용과 충분히 유사하지 않습니다. "
                         "파일이 잘못 업로드되었거나 코덱/형식 문제일 수 있습니다.")
                st.stop()
        except Exception:
            st.warning("전사 유사도 검증을 수행하지 못했습니다. 계속 진행합니다.")

        st.session_state.transcript = transcript

        with st.spinner("요약/분류 중…"):
            summary = summarize_transcript(transcript, api_key=(api_key or None))
            if isinstance(summary, str):
                summary = {"brief": summary, "bullets": [], "decisions": [], "actions": []}
            doc_type = decide_doc_type(summary)
            st.session_state.summary = summary
            st.session_state.doc_type = doc_type

        st.markdown("**전사 미리보기 (앞 400자)**")
        st.code(transcript[:400] + ("..." if len(transcript) > 400 else ""))

        st.success(f"문서 유형: {'연구노트' if doc_type=='research' else '일반 회의록'}")

        if not auto_detect:
            doc_type = st.radio("문서 유형 선택", options=["general","research"], index=0, horizontal=True)

        # 템플릿 컨텍스트
        if doc_type == "research":
            enrich = summary.get("research_enrich", {}) if isinstance(summary, dict) else {}
            context = {
                "meta": meta,
                "summary": {
                    "brief": summary.get("brief",""),
                    "bullets": summary.get("bullets",[]),
                    "decisions": summary.get("decisions",[]),
                    "actions": summary.get("actions",[]),
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
                    "brief": summary.get("brief",""),
                    "bullets": summary.get("bullets",[]),
                    "decisions": summary.get("decisions",[]),
                    "actions": summary.get("actions",[]),
                },
                "transcript": transcript
            }
            template_name = "meeting.md.j2"
            out_base = "회의록_" + dt.datetime.now().strftime("%Y%m%d_%H%M")

        md_text = render_markdown("templates", template_name, context)
        md_path = f"{out_base}.md"; pdf_path = f"{out_base}.pdf"

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
        st.code(md_text[:1500] + ("..." if len(md_text) > 1500 else ""), language="markdown")
else:
    st.info("좌측에서 오디오를 업로드하면 시작할 수 있어요.")
