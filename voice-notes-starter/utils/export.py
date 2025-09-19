# utils/export.py
from __future__ import annotations

# --- 표준 라이브러리 ---
from pathlib import Path
import os
import io
import shutil
import tempfile
import subprocess
from typing import Optional, List

# --- Jinja2 (템플릿) ---
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

# ---- (선택) Markdown → HTML 변환 라이브러리 (없으면 폴백)
try:
    import markdown as mdlib  # pip install markdown
except Exception:
    mdlib = None

# ---- (선택) HTML → PDF (wkhtmltopdf + pdfkit) 폴백 1
try:
    import pdfkit  # pip install pdfkit (wkhtmltopdf 필요)
except Exception:
    pdfkit = None

# ---- (선택) ReportLab PDF 작성 폴백 2
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import mm
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    reportlab_available = True
except Exception:
    reportlab_available = False


# --------------------------------------------------------------------------------------
# 내부 유틸
# --------------------------------------------------------------------------------------
def _template_dirs(hint: str | Path | None = None) -> List[str]:
    """
    Streamlit Cloud/로컬 어디서 실행하든 템플릿 폴더를 안정적으로 찾기 위한 후보 경로들.
    존재하는 디렉터리만 문자열 경로로 반환(중복 제거).
    """
    here = Path(__file__).resolve().parent          # .../utils
    repo = here.parent                               # 프로젝트 루트 가정

    cands: List[Path] = []
    if hint:
        p = Path(hint)
        cands += [p, here / p, repo / p]

    # 일반적인 위치들
    cands += [
        repo / "templates",
        here / "templates",
        Path.cwd() / "templates",
    ]

    # Streamlit Cloud에서 자주 보이는 경로(있으면 사용)
    cands += [
        Path("/mount/src/streamlit_app/voice-notes-starter/templates"),
        Path("/mount/src/app/templates"),
    ]

    seen: set[str] = set()
    out: List[str] = []
    for d in cands:
        try:
            r = d.resolve()
        except Exception:
            continue
        if r.is_dir():
            s = str(r)
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _ensure_utf8_no_bom(text: str) -> bytes:
    """
    안전한 UTF-8(쉼표 없음) 바이트로 변환.
    """
    if isinstance(text, bytes):
        # 이미 bytes면 그대로 사용하되, UTF-8로 재해석 시도는 하지 않음
        return text
    return text.encode("utf-8", "replace")


# --------------------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------------------
def render_markdown(templates_dir: str | Path | None, template_name: str, context: dict) -> str:
    """
    Jinja2 템플릿을 로드하여 Markdown 문자열로 렌더링.
    templates_dir는 'templates' 같은 힌트일 뿐이며, 실제로는 여러 후보 경로를 탐색.
    """
    dirs = _template_dirs(templates_dir)
    if not dirs:
        raise FileNotFoundError("템플릿 폴더를 찾을 수 없습니다. 레포에 'templates/' 폴더가 커밋되어 있는지 확인하세요.")

    env = Environment(
        loader=FileSystemLoader(dirs),
        autoescape=select_autoescape(enabled_extensions=("j2", "md", "html"))
    )

    try:
        tpl = env.get_template(template_name)  # 예: "meeting.md.j2"
    except TemplateNotFound as e:
        # 어떤 파일들이 보이는지 힌트 제공
        existing = []
        for d in dirs:
            existing += [p.name for p in Path(d).glob("*")]
        existing = sorted(set(existing))
        raise FileNotFoundError(
            f"TemplateNotFound: '{template_name}'. "
            f"탐색 경로: {dirs} | 발견된 파일: {existing}"
        ) from e

    text = tpl.render(**(context or {}))
    return text if isinstance(text, str) else text.decode("utf-8", "replace")


def save_markdown(md_text: str, out_dir: str | Path, filename: str = "document.md") -> Path:
    """
    Markdown 텍스트를 UTF-8로 저장하고, 저장된 경로를 반환.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_bytes(_ensure_utf8_no_bom(md_text))
    return out_path


def markdown_to_pdf(md_text: str, out_pdf_path: str | Path | None = None, html_title: str = "Document") -> Path:
    """
    Markdown → PDF 변환.
    - 1순위: pdfkit + wkhtmltopdf(설치되어 있을 때)
    - 2순위: ReportLab(간단한 문단 렌더링)
    - 최후: 아주 단순한 txt → PDF (ReportLab 캔버스)
    """
    # 출력 경로 준비
    if out_pdf_path is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="md2pdf_"))
        out_pdf_path = tmpdir / "document.pdf"
    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # HTML 변환(가능하면)
    html_str: Optional[str] = None
    if mdlib is not None:
        try:
            html_str = mdlib.markdown(md_text or "", extensions=["extra", "tables", "sane_lists"])
        except Exception:
            # markdown 실패 시 html_str=None 으로 두고 폴백
            html_str = None
    if html_str is None:
        # 마크다운 라이브러리가 없으면 최소한의 프리포맷 HTML로 감싸기
        from html import escape
        html_str = f"<html><head><meta charset='utf-8'><title>{escape(html_title)}</title></head><body><pre>{escape(md_text or '')}</pre></body></html>"

    # 1) pdfkit + wkhtmltopdf 시도
    if pdfkit is not None:
        try:
            # wkhtmltopdf 바이너리 확인(환경변수나 PATH에 있어야 함)
            config = None
            wkhtml = os.environ.get("WKHTMLTOPDF_BINARY")
            if wkhtml and Path(wkhtml).exists():
                config = pdfkit.configuration(wkhtmltopdf=wkhtml)  # 명시 설정
            # 임시 HTML 파일 생성 후 변환
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
                f.write(html_str)
                html_path = f.name
            try:
                pdfkit.from_file(html_path, str(out_pdf_path), configuration=config)
                return out_pdf_path
            finally:
                try:
                    os.unlink(html_path)
                except Exception:
                    pass
        except Exception:
            # 다음 폴백으로 진행
            pass

    # 2) ReportLab (문단 기반)
    if reportlab_available:
        try:
            styles = getSampleStyleSheet()
            style = styles["BodyText"]

            # 한글 폰트가 필요하면(옵션): 환경에 TTF가 있다면 등록
            # 예) NotoSansCJK-Regular.ttc / NanumGothic.ttf 경로가 있다면 등록해서 스타일에 적용 가능
            # 아래는 자동 탐색(성공하면 적용)
            def _try_register_korean_font():
                candidates = [
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
                    str(Path.home() / "fonts" / "NanumGothic.ttf"),
                ]
                for c in candidates:
                    p = Path(c)
                    if p.exists():
                        try:
                            pdfmetrics.registerFont(TTFont("KOREAN", str(p)))
                            return "KOREAN"
                        except Exception:
                            continue
                return None

            kfont = _try_register_korean_font()
            if kfont:
                style.fontName = kfont

            doc = SimpleDocTemplate(str(out_pdf_path), pagesize=A4, title=html_title)
            story = []
            for para in (md_text or "").split("\n\n"):
                # 매우 단순한 문단 나눔(***고급 마크다운 렌더링 아님***)
                story.append(Paragraph(para.replace("\n", "<br/>"), style))
                story.append(Spacer(1, 4 * mm))
            doc.build(story)
            return out_pdf_path
        except Exception:
            # 아래 최후 폴백으로 진행
            pass

    # 3) 최후 폴백: ReportLab 캔버스로 프리텍스트 느낌으로 저장(아주 단순)
    try:
        from reportlab.pdfgen import canvas  # 경량 import
        from reportlab.lib.pagesizes import A4

        c = canvas.Canvas(str(out_pdf_path), pagesize=A4)
        width, height = A4

        x_margin = 20 * mm
        y = height - 20 * mm
        line_height = 5 * mm

        # 기본 폰트(영문 위주). 한글은 폰트 등록이 필요하므로 위 2) 경로를 권장.
        c.setFont("Helvetica", 10)

        for line in (md_text or "").splitlines():
            if y < 20 * mm:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 20 * mm
            # 너무 긴 줄은 잘라서 표시(간단 처리)
            while len(line) > 100:
                c.drawString(x_margin, y, line[:100])
                line = line[100:]
                y -= line_height
                if y < 20 * mm:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 20 * mm
            c.drawString(x_margin, y, line)
            y -= line_height

        c.save()
        return out_pdf_path
    except Exception as e:
        raise RuntimeError(
            "Markdown을 PDF로 변환하지 못했습니다. pdfkit(wkhtmltopdf) 또는 reportlab 설치를 권장합니다."
        ) from e
