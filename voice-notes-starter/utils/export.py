from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, select_autoescape
import markdown as md
import pdfkit
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import shutil
import textwrap as _tw

__all__ = ["render_markdown", "save_markdown", "markdown_to_pdf"]

# ----------------------- 기본 템플릿 폴백 -----------------------
DEFAULT_TEMPLATES: Dict[str, str] = {
    "meeting.md.j2": """# {{ meta.title }}
- 일시: {{ meta.dt }}
- 장소: {{ meta.place }}
- 참석: {{ meta.attendees }}
- 진행: {{ meta.host }} / 서기: {{ meta.scribe }}

## 요약
{{ summary.brief }}

## 주요 논의
{% for b in summary.bullets -%}
- {{ b }}
{% endfor %}

## 의사결정
{% for d in summary.decisions -%}
- {{ d }}
{% endfor %}

## 액션아이템
{% for a in summary.actions -%}
- [ ] {{ a }}
{% endfor %}

---

## 원문 전사
{{ transcript }}
""",
    "research.md.j2": """# 연구노트: {{ meta.title }}
- 일시: {{ meta.dt }}
- 장소: {{ meta.place }}
- 연구원: {{ meta.attendees }}
- 진행: {{ meta.host }} / 서기: {{ meta.scribe }}

## 개요(Brief)
{{ summary.brief }}

## 목표(Objective)
{{ summary.objective }}

## 방법(Methods)
{% for m in summary.methods -%}
- {{ m }}
{% endfor %}

## 결과(Results)
{% for r in summary.results -%}
- {{ r }}
{% endfor %}

## 한계(Limitations)
{{ summary.limitations }}

## 의사결정/액션
{% for d in summary.decisions -%}
- 결정: {{ d }}
{% endfor %}
{% for a in summary.actions -%}
- [ ] {{ a }}
{% endfor %}

---

## 원문 전사
{{ transcript }}
"""
}

# ----------------------- HTML wrapper (wkhtmltopdf 용) -----------------------
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

# ----------------------- 템플릿 로딩 보강 -----------------------
def _candidate_template_dirs(templates_dir: str) -> List[str]:
    """여러 실행 환경에서 템플릿을 찾기 위한 후보 경로 생성"""
    here = os.path.dirname(os.path.abspath(__file__))           # utils 폴더
    cwd  = os.getcwd()                                          # 런타임 CWD
    project_root = os.path.abspath(os.path.join(here, ".."))    # 프로젝트 루트 추정
    parent_of_root = os.path.abspath(os.path.join(project_root, ".."))

    candidates = [
        os.path.abspath(templates_dir),                 # 호출부 기준 상대/절대
        os.path.join(cwd, templates_dir),               # CWD 기준
        os.path.join(project_root, templates_dir),      # 프로젝트 루트 기준
        os.path.join(parent_of_root, templates_dir),    # 한 단계 위(Cloud 일부 케이스)
    ]
    # 중복 제거 + 존재하는 디렉터리만
    out: List[str] = []
    seen = set()
    for p in candidates:
        if os.path.isdir(p) and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def render_markdown(template_dir: str, template_name: str, context: Dict[str, Any]) -> str:
    """
    templates_dir와 template_name을 받아 안전하게 템플릿을 로딩.
    - 여러 후보 경로를 순회
    - 없으면 DEFAULT_TEMPLATES 사용
    - 그래도 없으면 어떤 경로를 뒤졌는지 명시하며 예외
    """
    search_paths = _candidate_template_dirs(template_dir)
    env = Environment(
        loader=FileSystemLoader(search_paths),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),  # md.j2는 autoescape 비중요
        undefined=StrictUndefined
    )
    try:
        tpl = env.get_template(template_name)
        return tpl.render(**context)
    except TemplateNotFound:
        fallback = DEFAULT_TEMPLATES.get(template_name)
        if fallback:
            tmp_env = Environment(undefined=StrictUndefined, autoescape=False)
            tpl = tmp_env.from_string(fallback)
            return tpl.render(**context)
        raise FileNotFoundError(
            f"Template '{template_name}' not found. searched in: {search_paths}.\n"
            f"Place your templates at one of these directories or provide a fallback."
        )

# ----------------------- 파일 저장 -----------------------
def save_markdown(md_text: str, out_path: str):
    # Windows 메모장 호환: UTF-8 BOM으로 저장
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(md_text)

# ----------------------- ReportLab 폰트 등록 -----------------------
def _register_korean_font() -> Optional[str]:
    """
    ReportLab 폴백 PDF용: 시스템 한글 폰트를 찾아 등록.
    우선순위: Malgun Gothic -> Noto Sans KR -> NanumGothic
    (리눅스/Cloud에선 Noto/Nanum가 있을 가능성 높음)
    """
    candidates = [
        # Windows
        ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf"),
        ("NotoSansKR",   r"C:\Windows\Fonts\NotoSansKR-Regular.otf"),
        ("NotoSansKR",   r"C:\Windows\Fonts\NotoSansKR-Regular.ttf"),
        ("NanumGothic",  r"C:\Windows\Fonts\NanumGothic.ttf"),
        # Linux(common)
        ("NotoSansKR",   "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ("NotoSansKR",   "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf"),
        ("NanumGothic",  "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return None

# ----------------------- wkhtmltopdf 감지 -----------------------
def _detect_wkhtmltopdf() -> Optional[str]:
    """
    wkhtmltopdf 실행파일 위치 자동 감지.
    우선순위: 환경변수 -> PATH -> 일반 설치 경로(리눅스/맥/윈도우 추정)
    """
    # 0) 환경변수
    env_path = os.getenv("WKHTMLTOPDF_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # 1) PATH
    path = shutil.which("wkhtmltopdf")
    if path:
        return path

    # 2) 일반 위치들
    candidates = [
        "/usr/bin/wkhtmltopdf",
        "/usr/local/bin/wkhtmltopdf",
        "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
        "C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None

# ----------------------- Markdown → PDF -----------------------
def markdown_to_pdf(md_text: str, out_pdf: str) -> str:
    """
    1) wkhtmltopdf 있으면 HTML→PDFKit (UTF-8/한글 폰트 스타일 포함)
    2) 실패 시 ReportLab 폴백: 시스템 한글 폰트 등록 후 텍스트 렌더링
    """
    # 1) 고품질 경로 (wkhtmltopdf)
    try:
        html = md.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
        html = _html_wrap(html)

        wkhtml = _detect_wkhtmltopdf()
        if wkhtml:
            config = pdfkit.configuration(wkhtmltopdf=wkhtml)
        else:
            # config=None로 호출하면 PATH에 있어야만 동작.
            config = None

        options = {
            "encoding": "UTF-8",
            "quiet": ""
        }
        # 출력 디렉터리 없으면 생성
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        pdfkit.from_string(html, out_pdf, options=options, configuration=config)
        return out_pdf
    except Exception:
        # 2) 폴백: ReportLab (간이 레이아웃)
        kfont = _register_korean_font() or "Helvetica"
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        c = canvas.Canvas(out_pdf, pagesize=A4)
        width, height = A4
        x, y = 20 * mm, height - 20 * mm
        try:
            c.setFont(kfont, 11)
        except Exception:
            c.setFont("Helvetica", 11)

        max_chars = 78  # 간이 줄바꿈(페이지 폭에 맞게 조절)
        for line in md_text.splitlines():
            wrapped = _tw.wrap(
                line,
                width=max_chars,
                replace_whitespace=False,
                drop_whitespace=False
            )
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
