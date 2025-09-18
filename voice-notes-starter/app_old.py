\
import streamlit as st
from dotenv import load_dotenv
import os, io, datetime as dt
from utils.transcribe import transcribe_audio, get_audio_duration_seconds
from utils.summarize import summarize_transcript
from utils.classify import decide_doc_type
from utils.export import render_markdown, save_markdown, markdown_to_pdf

load_dotenv()

st.set_page_config(page_title="병원 회의용 음성 자동 노트", page_icon="🩺", layout="centered")

st.title("🩺 음성기반 자동 회의록/연구노트")
st.caption("업로드 → 전사 → 요약 → 서식 적용 → Markdown/PDF 저장")

# Sidebar: settings
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key (미입력 시 .env 사용)", type="password", value="")
    auto_detect = st.toggle("회의 타입 자동감지(일반/연구)", value=True)
    language_hint = st.selectbox("전사 언어 힌트", ["Auto", "ko", "en", "ja", "zh"], index=0)
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
        with st.spinner("전사 중…"):
            transcript = transcribe_audio(bytes_data, uploaded.name, api_key=(api_key or None), language_hint=None if language_hint=="Auto" else language_hint)

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
