# export.py
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown as md
import pdfkit
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import os
import textwrap as _tw
import platform

__all__ = ["render_markdown", "save_markdown", "markdown_to_pdf"]

# ---------- HTML wrapper for wkhtmltopdf ----------
def _html_wrap(markdown_html: str) -> str:
    # 시스템 한글 폰트 사용: Noto Sans CJK KR / Nanum Gothic / Malgun
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 20mm; }}
  body {{
    font-family: "Noto Sans CJK KR", "Noto Sans KR", "Nanum Gothic", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    font-size: 12pt;
    line-height: 1.5;
    word-break: keep-all;
  }}
  h1, h2, h3, h4, h5 {{ font-weight: 700; }}
  table {{
    border-collapse: collapse;
    width: 100%;
  }}
  th, td {{
    border: 1px solid #999;
    padding: 6px 8px;
    vertical-align: top;
  }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
</style>
</head>
<body>
{markdown_html}
</body>
</html>"""
# --------------------------------------------------

# 템플릿 디렉토리 절대 경로 고정
ROOT_DIR = Path(__file__).resolve().parents[1]      # .../voice-notes-starter
TEMPLATES_DIR = ROOT_DIR / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml", "md", "jinja", "j2"])
)

def render_markdown(template_name: str, context: Dict[str, Any]) -> str:
    tpl = _env.get_template(template_name)
    return tpl.render(**context)

def save_markdown(md_text: str, out_path: str):
    # Windows 메모장 호환: UTF-8 BOM
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(md_text)

def _register_korean_font() -> Optional[str]:
    """
    ReportLab 폴백용 시스템 한글폰트 등록.
    OS별 후보 경로를 탐색해서 먼저 발견되는 폰트를 등록합니다.
    """
    candidates: list[tuple[str, str]] = []

    if platform.system() == "Windows":
        candidates += [
            ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf"),
            ("NanumGothic",  r"C:\Windows\Fonts\NanumGothic.ttf"),
            ("NotoSansKR",   r"C:\Windows\Fonts\NotoSansKR-Regular.otf"),
            ("NotoSansKR",   r"C:\Windows\Fonts\NotoSansKR-Regular.ttf"),
        ]
    else:
        # Ubuntu (Streamlit Cloud)
        candidates += [
            ("NotoSansCJKkr", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            ("NotoSansCJKkr", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            ("NotoSansKR",    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf"),
            ("NotoSansKR",    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf"),
            ("NanumGothic",   "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ]

    # 추가로 폴더 전체를 훑고 싶다면 여기에 glob를 넣을 수도 있음

    for name, path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return None

def markdown_to_pdf(md_text: str, out_pdf: str) -> str:
    """
    1) wkhtmltopdf 있으면 HTML→PDFKit (시스템 한글폰트 사용)
    2) 실패 시 ReportLab 폴백: 시스템 한글폰트 등록 후 단순 렌더링
    """
    # 1) 고품질 경로 (웹폰트 없이 시스템 폰트 사용)
    try:
        html = md.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
        html = _html_wrap(html)

        options = {
            "encoding": "UTF-8",
            "quiet": "",
            # wkhtmltopdf가 로컬 리소스 접근할 수 있도록 (일부 환경 호환)
            "enable-local-file-access": None,
            # DPI/텍스트 렌더링 품질을 높이고 싶으면 아래 옵션도 가능
            # "dpi": 96,
            # "print-media-type": None,
        }
        pdfkit.from_string(html, out_pdf, options=options)  # wkhtmltopdf 필요
        return out_pdf
    except Exception:
        # 2) 폴백: ReportLab
        kfont = _register_korean_font() or "Helvetica"
        c = canvas.Canvas(out_pdf, pagesize=A4)
        width, height = A4
        x, y = 20 * mm, height - 20 * mm
        try:
            c.setFont(kfont, 11)
        except Exception:
            c.setFont("Helvetica", 11)

        max_chars = 78  # 간이 줄바꿈
        for line in md_text.splitlines():
            wrapped = _tw.wrap(line, width=max_chars, replace_whitespace=False, drop_whitespace=False)
            if not wrapped:
                wrapped = [""]
            for chunk in wrapped
