# utils/export.py
from __future__ import annotations

# --- 표준 라이브러리 ---
from pathlib import Path
import os
import tempfile
from typing import Optional, List, Union

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
def _template_dirs(hint: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Streamlit Cloud/로컬 어디서 실행하든 템플릿 폴더를 안정적으로 찾기 위한 후보 경로들.
    존재하는 디렉터리만 문자열 경로로 반환(중복 제거).
    """
    here = Path(__file__).resolve().parent          # .../utils
    repo = here.parent                               # 프로젝트 루트 가정

    cands = []  # type: List[Path]
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

    seen = set()     # type: set
    out = []         # type: List[str]
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


def _ensure_utf8_no_bom(text):  # type: (Union[str, bytes]) -> bytes
    """
    안전한 UTF-8 바이트로 변환(BOM 없음).
    """
    if isinstance(text, bytes):
        return text
    return text.encode("utf-8", "replace")


def _find_korean_font_path() -> Optional[Path]:
    """
    레포 동봉 폰트 및 시스템 폰트를 순차 탐색.
    사용자가 제공한 H2GTRM.TTF를 최우선으로 시도.
    """
    utils_dir = Path(__file__).resolve().parent
    repo_root = utils_dir.parent

    bundled_fonts = [
        repo_root / "fonts" / "H2GTRM.TTF",           # 사용자가 넣어둔 폰트(최우선)
        repo_root / "fonts" / "NanumGothic.ttf",
        repo_root / "fonts" / "NotoSansCJK-Regular.ttf",
        repo_root / "fonts" / "NotoSansKR-Regular.otf",
    ]
    system_fonts = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf"),
    ]

    for p in bundled_fonts + system_fonts:
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------------------
def render_markdown(templates_dir: Optional[Union[str, Path]],
                    template_name: str,
                    context: dict) -> str:
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
            "TemplateNotFound: '{}'.".format(template_name) +
            " 탐색 경로: {} | 발견된 파일: {}".format(dirs, existing)
        ) from e

    text = tpl.render(**(context or {}))
    return text if isinstance(text, str) else text.decode("utf-8", "replace")


def save_markdown(md_text: str, out_dir: Union[str, Path], filename: str = "document.md") -> Path:
    """
    Markdown 텍스트를 UTF-8로 저장하고, 저장된 경로를 반환.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_bytes(_ensure_utf8_no_bom(md_text))
    return out_path


def markdown_to_pdf(md_text: str,
                    out_pdf_path: Optional[Union[str, Path]] = None,
                    html_title: str = "Document") -> Path:
    """
    Markdown → PDF 변환 (한글 폰트 임베드 대응)
    - 1순위: pdfkit + wkhtmltopdf (CSS @font-face + enable-local-file-access)
    - 2순위: ReportLab (TTFont 등록 후 문단 렌더)
    - 최후: canvas 폴백
    """
    # 출력 경로 준비
    if out_pdf_path is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="md2pdf_"))
        out_pdf_path = tmpdir / "document.pdf"
    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # 0) 폰트 경로 탐색
    font_path = _find_korean_font_path()
    font_family_name = "DocKorean"  # CSS/ReportLab에서 사용할 논리 이름

    # 1) Markdown → HTML
    if mdlib is not None:
        try:
            html_body = mdlib.markdown(md_text or "", extensions=["extra", "tables", "sane_lists"])
        except Exception:
            from html import escape
            html_body = "<pre>{}</pre>".format(escape(md_text or ""))
    else:
        from html import escape
        html_body = "<pre>{}</pre>".format(escape(md_text or ""))

    # 2) CSS 구성 (폰트 임베드)
    base_css = [
        "html,body{margin:24px;font-size:14px;line-height:1.6;}",
        "table{border-collapse:collapse;} td,th{border:1px solid #888;padding:6px;}",
        "code,pre{font-family:monospace;}",
    ]
    font_css = ""
    if font_path:
        # wkhtmltopdf가 로컬 파일을 읽어올 수 있도록 file:// 스킴 사용
        font_url = "file://" + str(font_path.resolve()).replace("\\", "/")
        font_css = (
            "@font-face{font-family:'" + font_family_name + "';"
            "src:url('" + font_url + "') format('truetype');"
            "font-weight:normal;font-style:normal;}"
            "body,p,li,td,th,h1,h2,h3,h4,h5,h6{font-family:'" + font_family_name + "','DejaVu Sans',Arial,sans-serif;}"
        )
    else:
        base_css.append("body{font-family:'DejaVu Sans', Arial, sans-serif;}")

    html_str = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'/>"
        "<title>{}</title>"
        "<style>{}\n{}</style>"
        "</head><body>{}</body></html>"
    ).format(html_title, font_css, "\n".join(base_css), html_body)

    # 3) pdfkit + wkhtmltopdf (권장 경로)
    if pdfkit is not None:
        try:
            config = None
            wkhtml = os.environ.get("WKHTMLTOPDF_BINARY")
            if wkhtml and Path(wkhtml).exists():
                config = pdfkit.configuration(wkhtmltopdf=wkhtml)

            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
                f.write(html_str)
                html_path = f.name

            options = {
                # 한글 폰트/이미지 등 로컬 리소스 접근 허용 (매우 중요)
                "enable-local-file-access": None,
                "encoding": "UTF-8",
                # 페이지 옵션
                "margin-top": "10mm",
                "margin-bottom": "12mm",
                "margin-left": "10mm",
                "margin-right": "10mm",
                "page-size": "A4",
            }

            try:
                pdfkit.from_file(html_path, str(out_pdf_path), configuration=config, options=options)
                return out_pdf_path
            finally:
                try:
                    os.unlink(html_path)
                except Exception:
                    pass
        except Exception:
            # 다음 폴백
            pass

    # 4) ReportLab 경로 (폰트 등록 후 문단 렌더)
    if reportlab_available:
        try:
            styles = getSampleStyleSheet()
            style = styles["BodyText"]

            if font_path and font_path.exists():
                try:
                    pdfmetrics.registerFont(TTFont(font_family_name, str(font_path)))
                    style.fontName = font_family_name
                except Exception:
                    # 등록 실패 시 기본값 유지
                    pass

            doc = SimpleDocTemplate(str(out_pdf_path), pagesize=A4, title=html_title)
            story = []
            for para in (md_text or "").split("\n\n"):
                story.append(Paragraph(para.replace("\n", "<br/>"), style))
                story.append(Spacer(1, 4 * mm))
            doc.build(story)
            return out_pdf_path
        except Exception:
            # 최후 폴백
            pass

    # 5) 최후 폴백: 아주 단순한 canvas 출력 (한글 폰트 등록 시에만 권장)
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4

        c = canvas.Canvas(str(out_pdf_path), pagesize=A4)
        width, height = A4
        x_margin = 20 * mm
        y = height - 20 * mm
        line_height = 5 * mm

        # 폰트 등록 (가능하면)
        if reportlab_available and font_path and font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont(font_family_name, str(font_path)))
                c.setFont(font_family_name, 10)
            except Exception:
                c.setFont("Helvetica", 10)  # 한글 미지원
        else:
            c.setFont("Helvetica", 10)      # 한글 미지원

        for line in (md_text or "").splitlines():
            if y < 20 * mm:
                c.showPage()
                try:
                    c.setFont(font_family_name, 10)
                except Exception:
                    c.setFont("Helvetica", 10)
                y = height - 20 * mm

            # 너무 긴 줄은 단순 분할
            while len(line) > 90:
                c.drawString(x_margin, y, line[:90])
                line = line[90:]
                y -= line_height
                if y < 20 * mm:
                    c.showPage()
                    try:
                        c.setFont(font_family_name, 10)
                    except Exception:
                        c.setFont("Helvetica", 10)
                    y = height - 20 * mm

            c.drawString(x_margin, y, line)
            y -= line_height

        c.save()
        return out_pdf_path
    except Exception as e:
        raise RuntimeError(
            "Markdown을 PDF로 변환하지 못했습니다. wkhtmltopdf 또는 reportlab 설치/폰트 동봉을 권장합니다."
        ) from e
