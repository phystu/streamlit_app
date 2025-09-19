# utils/export.py  ← 전체 교체
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown as md
import pdfkit
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import textwrap as _tw

__all__ = ["render_markdown", "save_markdown", "markdown_to_pdf"]

# ---------- HTML wrapper for wkhtmltopdf ----------
def _html_wrap(markdown_html: str) -> str:
    # UTF-8 + 한글 폰트 지정
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 20mm; }}
  body {{
    font-family: "Malgun Gothic", "Noto Sans KR", "Apple SD Gothic Neo", "Nanum Gothic", sans-serif;
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

def render_markdown(template_dir: str, template_name: str, context: Dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape()
    )
    tpl = env.get_template(template_name)
    return tpl.render(**context)

def save_markdown(md_text: str, out_path: str):
    # Windows 메모장 호환: UTF-8 BOM으로 저장
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(md_text)

def _register_korean_font() -> Optional[str]:
    """
    ReportLab 폴백 PDF용: 시스템 한글 폰트를 찾아 등록.
    우선순위: Malgun Gothic -> Noto Sans KR -> NanumGothic
    """
    candidates = [
        ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf"),
        ("NotoSansKR",   r"C:\Windows\Fonts\NotoSansKR-Regular.otf"),
        ("NotoSansKR",   r"C:\Windows\Fonts\NotoSansKR-Regular.ttf"),
        ("NanumGothic",  r"C:\Windows\Fonts\NanumGothic.ttf"),
    ]
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
    1) wkhtmltopdf 있으면 HTML→PDFKit (UTF-8/한글 폰트 스타일 포함)
    2) 실패 시 ReportLab 폴백: 시스템 한글 폰트 등록 후 텍스트 렌더링
    """
    # 1) 고품질 경로
    try:
        html = md.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
        html = _html_wrap(html)
        options = {"encoding": "UTF-8", "quiet": ""}
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
            for chunk in wrapped:
                c.drawString(x, y, chunk)
                y -= 6 * mm
                if y < 20 * mm:
                    c.showPage()
                    try:
                        c.setFont(kfont, 11)
                    except Exception:
                        c.setFont("Helvetica", 11)
                    y = height - 20 * mm
        c.save()
        return out_pdf
