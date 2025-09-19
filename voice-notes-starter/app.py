# app.py
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
import os, io, datetime as dt, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

# 내부 유틸
from utils.transcribe import transcribe_audio, get_audio_duration_seconds
from utils.summarize import summarize_transcript
from utils.classify import decide_doc_type
from utils.export import render_markdown, save_markdown, markdown_to_pdf

# ------------------------------------------------------------------------------
# 초기 설정
# ------------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="병원 회의용 음성 자동 노트", page_icon="🩺", layout="centered")

st.title("🩺 음성기반 자동 회의록/연구노트")
st.caption("업로드 → 전사 → 요약 → 서식 적용 → Markdown/PDF 저장")

# ------------------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------------------
def safe_slug(s: str, default: str = "note") -> str:
    s = (s or "").strip()
    if not s:
        s = default
    # 파일/HTTP 헤더 안전용 ASCII 슬러그
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120] or default

def fast_transcribe_ko_with_progress(
    bytes_data: bytes,
    filename: str,
    api_key: str | None,
    chunk_ms: int = 60_000,
    max_workers: int = 6
) -> str:
    """
    - 한국어 고정 전사
    - 16kHz 모노로 다운샘플 후 60초 단위 청크 분할
    - 각 청크를 32kbps mp3로 인메모리 인코딩해 API로 전송
    - 병렬 전사 + 진행률바 업데이트
    """
    # 1) 로드 & 다운샘플
    audio = AudioSegment.from_file(io.BytesIO(bytes_data))
    audio = audio.set_frame_rate(16_000).set_channels(1)

    # 2) 분할
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
    done_count = 0

    # 4) per-chunk 전사 함수
    def transcribe_one(idx_seg):
        idx, seg = idx_seg
        buf = io.BytesIO()
        seg.export(buf, format="mp3", bitrate="32k")  # 업로드 시간 단축
        data = buf.getvalue()
        # 멀티파트 헤더 ASCII 문제 회피: 전송용 파일명 고정
        out = transcribe_audio(data, f"chunk_{idx}.mp3", api_key=(api_key or None), language_hint="ko")
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

    # 6) 결합
    full_text = " ".join(t for t in texts if t)
    return full_text

# ------------------------------------------------------------------------------
# 사이드바 설정
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key (미입력 시 .env 사용)", type="password", value="")
    auto_detect = st.toggle("회의 타입 자동감지(일반/연구)", value=True)
    st.markdown("---")
    st.markdown("**PDF 엔진 메모**  \n우선 `wkhtmltopdf`가 있으면 고품질 PDF, 없으면 단순 PDF로 저장합니다.")

# ------------------------------------------------------------------------------
# 업로드 & 메타
# ------------------------------------------------------------------------------
st.subheader("1) 오디오 업로드")
uploaded = st.file_uploader("오디오 파일 (mp3, m4a, wav, ogg, webm 등)",
                            type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"])

st.subheader("2) 기본데이터")
col1, col2 = st.columns(2)
with col1:
    mt_title = st.text_input("제목", "의생명연구원 정기회의")
    mt_dt = st.text_input("일시", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    mt_place = st.text_input("장소", "의생명연구원 2층 회의실")
with col2:
    mt_att = st.text_input("참석자(역할)", "홍길동(PI), 김철수(기획), 박영희(연구)…")
    host = st.text_input("진행/서기", "진행: 홍길동 / 서기: 김철수")

meta = {
    "title": mt_title,
    "dt": mt_dt,
    "place": mt_place,
    "attendees": mt_att,
    "host": (host.split("/")[0].replace("진행:", "").strip() if host else ""),
    "scribe": (host.split("/")[-1].replace("서기:", "").strip() if host else ""),
    "project": ""
}

st.divider()

# ------------------------------------------------------------------------------
# 메인 동작
# ------------------------------------------------------------------------------
if uploaded is not None:
    ext = uploaded.name.split(".")[-1].lower()
    bytes_data = uploaded.getvalue()

    # 길이 체크 (데모: 최대 2시간)
    try:
        duration = get_audio_duration_seconds(bytes_data, ext)
        st.write(f"오디오 길이: {duration/60:.1f}분")
        if duration > 2 * 3600:
            st.error("데모 제한: 2시간을 초과한 파일은 처리하지 않습니다.")
            st.stop()
    except Exception as e:
        st.warning(f"길이 확인 실패(계속 진행 가능): {e}")

    if st.button("전사 → 요약 → 서식 적용 실행", type="primary"):
        # -------- 전사 단계 --------
        with st.spinner("전사 중…"):
            transcript = fast_transcribe_ko_with_progress(
                bytes_data=bytes_data,
                filename=uploaded.name,
                api_key=(api_key or None),
                chunk_ms=60_000,       # 60초
                max_workers=6
            )

        # -------- 요약/분류 단계 --------
        with st.spinner("요약/분류 중…"):
            summary = summarize_transcript(transcript, api_key=(api_key or None))
            doc_type = decide_doc_type(summary)

        if not auto_detect:
            doc_type = st.radio("문서 유형 선택", options=["general", "research"], index=0, horizontal=True)

        st.success(f"문서 유형: {'연구노트' if doc_type=='research' else '일반 회의록'}")

        # -------- 템플릿/컨텍스트 --------
        if doc_type == "research":
            enrich = summary.get("research_enrich", {})
            context = {
                "meta": meta,
                "summary": {
                    "brief": summary.get("brief", ""),
                    "bullets": summary.get("bullets", []),
                    "decisions": summary.get("decisions", []),
                    "actions": summary.get("actions", []),
                    "objective": enrich.get("objective", ""),
                    "methods": enrich.get("methods", []),
                    "results": enrich.get("results", []),
                    "limitations": enrich.get("limitations", ""),
                },
                "transcript": transcript
            }
            template_name = "research.md.j2"
        else:
            context = {
                "meta": meta,
                "summary": {
                    "brief": summary.get("brief", ""),
                    "bullets": summary.get("bullets", []),
                    "decisions": summary.get("decisions", []),
                    "actions": summary.get("actions", []),
                },
                "transcript": transcript
            }
            template_name = "meeting.md.j2"

        # -------- Markdown 렌더링 --------
        md_text = render_markdown("templates", template_name, context)

        # -------- 출력 이름/폴더 --------
        default_base = ("연구노트_" if doc_type == "research" else "회의록_") + dt.datetime.now().strftime("%Y%m%d_%H%M")
        file_slug = safe_slug(meta.get("title") or default_base)
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        md_filename = f"{file_slug}.md"
        pdf_filename = f"{file_slug}.pdf"

        # -------- Markdown 즉시 다운로드(메모리) --------
        st.download_button(
            "📥 Markdown 다운로드",
            data=md_text.encode("utf-8"),
            file_name=md_filename,
            mime="text/markdown",
        )

        # (옵션) 디스크에도 저장하고 싶다면:
        md_file_path = save_markdown(md_text, out_dir, md_filename)

        # -------- PDF 생성 및 다운로드 --------
        try:
            pdf_path = markdown_to_pdf(md_text, out_pdf_path=out_dir / pdf_filename)  # Path 반환(또는 문자열)
            pdf_path = Path(pdf_path)
            if pdf_path.is_file():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 PDF 다운로드",
                        data=f,
                        file_name=pdf_filename,
                        mime="application/pdf",
                    )
            else:
                st.error("PDF 생성에 실패했습니다. logs를 확인하세요.")
        except Exception as e:
            st.error(f"PDF 변환 중 오류: {e}")

        # -------- 미리보기 --------
        st.markdown("미리보기(요약)")
        st.code(md_text[:1500] + ("..." if len(md_text) > 1500 else ""), language="markdown")

else:
    st.info("오디오 파일을 업로드하면 처리를 시작할 수 있어요.")

# ------------------------------------------------------------------------------
# (선택) 디버그 도구
# ------------------------------------------------------------------------------
#with st.expander("🛠 디버그(Cloud 경로/템플릿 확인)"):
#    st.write("CWD:", os.getcwd())
 #   try:
  #      st.write("프로젝트 루트 파일 목록:", os.listdir("."))
   #     st.write("utils 폴더:", os.listdir("utils"))
    #    st.write("templates 폴더:", os.listdir("templates") if Path("templates").exists() else "없음")
    #    st.write("outputs 폴더:", os.listdir("outputs") if Path("outputs").exists() else "없음")
   # except Exception as e:
    #    st.write("디버그 중 오류:", e)
