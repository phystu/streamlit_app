import streamlit as st
from dotenv import load_dotenv
import os, io, datetime as dt
from utils.transcribe import transcribe_audio, get_audio_duration_seconds
from utils.summarize import summarize_transcript
from utils.classify import decide_doc_type
from utils.export import render_markdown, save_markdown, markdown_to_pdf

# 추가: 고속 전사를 위한 라이브러리
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

load_dotenv()

st.set_page_config(page_title="병원 회의용 음성 자동 노트", page_icon="🩺", layout="centered")

st.title("🩺 음성기반 자동 회의록/연구노트")
st.caption("업로드 → 전사 → 요약 → 서식 적용 → Markdown/PDF 저장")

# ------------------------------
# 고속 전사(병렬 청크) 헬퍼 함수
# ------------------------------
def fast_transcribe_ko_with_progress(bytes_data: bytes, filename: str, api_key: str | None, chunk_ms: int = 60_000, max_workers: int = 6) -> str:
    """
    - 한국어 고정 전사
    - 16kHz 모노로 다운샘플 후 60초 단위 청크 분할
    - 각 청크를 32kbps mp3로 인메모리 인코딩해 API로 전송
    - 병렬 전사 + 진행률바 업데이트
    """
    # 1) 로드 & 다운샘플(속도 최적화)
    audio = AudioSegment.from_file(io.BytesIO(bytes_data))
    audio = audio.set_frame_rate(16_000).set_channels(1)

    # 2) 60초 청크 분할
    total_ms = len(audio)
    chunks = []
    starts = range(0, total_ms, chunk_ms)
    for i, start in enumerate(starts):
        seg = audio[start: start + chunk_ms]
        chunks.append((i, seg))

    total = len(chunks)
    if total == 0:
        return ""

    # 3) 진행률 UI
    progress = st.progress(0)
    status = st.empty()
    done_count = 0

    # 4) per-chunk 전사 함수
    def transcribe_one(idx_seg):
        idx, seg = idx_seg
        buf = io.BytesIO()
        # 업로드 시간 단축: 저비트레이트 mp3
        seg.export(buf, format="mp3", bitrate="32k")
        data = buf.getvalue()
        # utils.transcribe를 그대로 재사용(언어 힌트 한국어 고정)
        out = transcribe_audio(data, f"{idx}_{filename}.mp3", api_key=(api_key or None), language_hint="ko")
        # utils.transcribe가 문자열 또는 dict를 반환하는 경우 모두 대응
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
            done_count += 1
            pct = int(done_count / total * 100)
            progress.progress(pct)
            status.write(f"전사 진행률: {done_count}/{total} 청크 완료 ({pct}%)")

    progress.empty()
    status.empty()

    # 6) 원래 순서대로 결합
    full_text = " ".join(t for t in texts if t)
    return full_text

# Sidebar: settings
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key (미입력 시 .env 사용)", type="password", value="")
    auto_detect = st.toggle("회의 타입 자동감지(일반/연구)", value=True)

    # 🔒 한국어 전사 고정: 선택 UI 제거(원본 유지 원하면 주석 해제)
    # language_hint = st.selectbox("전사 언어 힌트", ["Auto", "ko", "en", "ja", "zh"], index=0)

    st.markdown("---")
    st.markdown("**PDF 엔진 메모**  \n우선 `wkhtmltopdf`가 있으면 고품질 PDF, 없으면 단순 PDF로 저장합니다.")

st.subheader("1) 오디오 업로드")
uploaded = st.file_uploader("오디오 파일 (mp3, m4a, wav, ogg, webm 등)", type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"])

st.subheader("2) 메타데이터")
col1, col2 = st.columns(2)
with col1:
    mt_title = st.text_input("제목", "의료혁신센터 정기회의")
    mt_dt = st.text_input("일시", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    mt_place = st.text_input("장소", "회의실 A")
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
    ext = uploaded.name.split(".")[-1].lower()
    bytes_data = uploaded.getvalue()

    # Duration check (max 2h for demo)
    try:
        duration = get_audio_duration_seconds(bytes_data, ext)
        st.write(f"오디오 길이: {duration/60:.1f}분")
        if duration > 2 * 3600:
            st.error("데모 제한: 2시간을 초과한 파일은 처리하지 않습니다.")
            st.stop()
    except Exception as e:
        st.warning(f"길이 확인 실패(계속 진행 가능): {e}")

    if st.button("전사 → 요약 → 서식 적용 실행", type="primary"):
        # -------- 전사 단계 (한국어 고정 + 진행률바 + 고속전사) --------
        with st.spinner("전사 중…"):
            # 기존 단일 호출 대신 병렬 청크 전사 사용
            transcript = fast_transcribe_ko_with_progress(
                bytes_data=bytes_data,
                filename=uploaded.name,
                api_key=(api_key or None),
                chunk_ms=60_000,      # 60초 단위
                max_workers=6         # 동시 요청 수(네트워크/요금 상황 따라 조절)
            )

        # -------- 요약/분류 단계 --------
        with st.spinner("요약/분류 중…"):
            summary = summarize_transcript(transcript, api_key=(api_key or None))
            doc_type = decide_doc_type(summary)

        if not auto_detect:
            doc_type = st.radio("문서 유형 선택", options=["general","research"], index=0, horizontal=True)

        st.success(f"문서 유형: {'연구노트' if doc_type=='research' else '일반 회의록'}")

        # Build context for templates
        if doc_type == "research":
            enrich = summary.get("research_enrich", {})
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
    st.info("오디오 파일을 업로드하면 처리를 시작할 수 있어요.")
