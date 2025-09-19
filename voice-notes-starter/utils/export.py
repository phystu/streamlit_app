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
# utils/export.py
from __future__ import annotations
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
from pathlib import Path

def _template_dirs(hint: str | None = None) -> list[str]:
    here = Path(__file__).resolve().parent        # .../utils
    repo = here.parent                             # 프로젝트 루트
    cands = []

    # 사용자가 넘긴 힌트 우선
    if hint:
        p = Path(hint)
        cands += [p, here / p, repo / p]

    # 일반적인 위치들
    cands += [
        repo / "templates",
        here / "templates",
        Path.cwd() / "templates",
        # Streamlit Cloud에서 흔한 경로들(있으면 사용)
        Path("/mount/src/streamlit_app/voice-notes-starter/templates"),
        Path("/mount/src/app/templates"),
    ]

    # 존재하는 디렉터리만 문자열 경로로 반환(중복 제거)
    seen, out = set(), []
    for d in cands:
        try:
            r = str(d.resolve())
        except Exception:
            continue
        if (d.is_dir()) and (r not in seen):
            out.append(r); seen.add(r)
    return out

def render_markdown(templates_dir: str | None, template_name: str, context: dict) -> str:
    dirs = _template_dirs(templates_dir)
    if not dirs:
        raise FileNotFoundError("템플릿 폴더를 찾을 수 없습니다. 레포에 templates/가 커밋되어 있는지 확인하세요.")

    env = Environment(
        loader=FileSystemLoader(dirs),
        autoescape=select_autoescape(enabled_extensions=("j2", "md", "html"))
    )

    try:
        tpl = env.get_template(template_name)      # 예: "meeting.md.j2"
    except TemplateNotFound as e:
        # 어떤 파일들이 보이는지 힌트 제공
        existing = []
        for d in dirs:
            existing += [p.name for p in Path(d).glob("*")]
        raise FileNotFoundError(
            f"TemplateNotFound: '{template_name}'. "
            f"탐색 경로: {dirs} | 발견된 파일: {sorted(set(existing))}"
        ) from e

    text = tpl.render(**(context or {}))
    return text if isinstance(text, str) else text.decode("utf-8", "replace")


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
