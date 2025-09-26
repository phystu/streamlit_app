# utils/summarize.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import os, json, re, datetime as dt

# --------------------------------------------------------------------
# System & Prompts
# --------------------------------------------------------------------
SYSTEM = (
    "You are a Korean note-taking assistant for hospital meetings. "
    "Reply ONLY with valid JSON (UTF-8). No code fences, no prose."
)

SUMMARY_PROMPT = """
다음은 병원 회의의 전사본이야. 핵심을 한국어로 간결하게 정리해.
응답은 JSON으로만 하고, 키는
- brief: 4-5문장 요약
- bullets: 5개 내외 핵심 bullet 리스트
- decisions: 회의에서 결정된 사항 bullet
- actions: 액션 배열 (각 항목은 {owner, task, due})
- type_hint: 'research' 또는 'general' 중 하나 (가능성 판단)

규칙(매우 중요):
- actions[].due 는 반드시 'YYYY-MM-DD' 형식(ISO, Asia/Seoul 기준 가정)으로만 출력. 모르면 null.
- 상대표현(예: '다음주 금요일', '말일')은 meeting_date를 기준으로 실제 날짜로 변환. meeting_date 없으면 null.
- 마감일은 meeting_date보다 과거일 수 없음. 과거가 되면 null.
- 임의 추정/환상 금지. 전사에 근거 없으면 null.
- actions 항목의 키는 owner, task, due 만 허용.
- bullets/decisions는 간결한 한국어 문장형 bullet로.

전사:
{{transcript}}
"""

RESEARCH_ENRICH = """
위 전사가 '연구회의'일 가능성이 높다면 연구노트용 보조 요약을 생성하라.
응답은 JSON만. 키:
- objective: 연구 배경/목표 2-3문장
- methods: 방법 bullet 3-6개
- results: 관찰/결과 bullet 3-6개 (수치가 있으면 보존)
- limitations: 한계/주의사항 1-3문장
- actions: 액션 배열 (각 항목은 {owner, task, due})

규칙(매우 중요):
- actions[].due 는 반드시 'YYYY-MM-DD' 형식(ISO, Asia/Seoul 기준 가정)으로만 출력. 모르면 null.
- 상대표현은 meeting_date를 기준으로 실제 날짜로 변환. meeting_date 없으면 null.
- 마감일은 meeting_date보다 과거일 수 없음. 과거가 되면 null.
- 임의 추정 금지. 전사/요약에 근거 없으면 null.
- actions 항목의 키는 owner, task, due 만 허용.
"""

# --------------------------------------------------------------------
# OpenAI Client
# --------------------------------------------------------------------
def get_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)

# --------------------------------------------------------------------
# Robust JSON loader
# --------------------------------------------------------------------
def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError(
        "JSON parsing failed",
        s[:200] + ("..." if len(s) > 200 else ""),
        0,
    )

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _is_valid_iso_date(s: Any) -> bool:
    if not isinstance(s, str) or not _ISO_DATE.match(s):
        return False
    try:
        dt.date.fromisoformat(s)
        return True
    except ValueError:
        return False

def _normalize_meeting_date(value: Optional[str]) -> Optional[str]:
    """
    다양한 포맷(한국식/숫자/구분자/시각 포함)을 ISO(YYYY-MM-DD)로 정규화.
    2자리 연도는 2000+yy로 해석.
    """
    if not value:
        return None
    s = str(value).strip()

    if _is_valid_iso_date(s):
        return s

    # '2025년 9월 22일' / '25년 9월 22일'
    m = re.search(r"^\s*(\d{2,4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        yy, mm, dd = m.groups()
        y = int(yy)
        if y < 100:
            y += 2000
        return f"{y:04d}-{int(mm):02d}-{int(dd):02d}"

    # '2025.9.22', '2025/09/22', '2025-09-22'
    m = re.search(r"^\s*(\d{2,4})[./-](\d{1,2})[./-](\d{1,2})", s)
    if m:
        yy, mm, dd = m.groups()
        y = int(yy)
        if y < 100:
            y += 2000
        return f"{y:04d}-{int(mm):02d}-{int(dd):02d}"

    # YYMMDD / YYYYMMDD
    m = re.search(r"^\s*(\d{6}|\d{8})\s*$", s)
    if m:
        raw = m.group(1)
        if len(raw) == 6:
            y = 2000 + int(raw[0:2])
            return f"{y:04d}-{int(raw[2:4]):02d}-{int(raw[4:6]):02d}"
        else:
            return f"{int(raw[0:4]):04d}-{int(raw[4:6]):02d}-{int(raw[6:8]):02d}"

    # 일반 텍스트에서 날짜만 추출 (시간/요일 무시)
    m = re.search(r"(?P<y>\d{2,4})[년./-]\s*(?P<m>\d{1,2})[월./-]\s*(?P<d>\d{1,2})", s)
    if m:
        y = int(m.group("y"))
        if y < 100:
            y += 2000
        return f"{y:04d}-{int(m.group('m')):02d}-{int(m.group('d')):02d}"

    return None

def _extract_meeting_date_from_meta(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    meta(dict)에서 회의일 후보 키를 찾아 정규화.
    우선순위: dt > date > meeting_date > '일시' > '날짜' > '일자'
    """
    if not isinstance(meta, dict):
        return None
    candidates = []
    for k in ("dt", "date", "meeting_date", "일시", "날짜", "일자"):
        if k in meta and meta[k]:
            candidates.append(str(meta[k]))
    for raw in candidates:
        iso = _normalize_meeting_date(raw)
        if iso:
            return iso
    return None

def _extract_meeting_date_from_text(transcript: str) -> Optional[str]:
    """
    전사 헤더에서 '일시:' 또는 date/날짜/일자 라인을 찾아 날짜를 정규화.
    마크다운 굵게(**일시:** ...) 형식도 지원.
    """
    if not transcript:
        return None

    m = re.search(r"(?:\*\*)?\s*일시\s*(?:\*\*)?\s*:\s*([^\n\r]+)", transcript, flags=re.I)
    if m:
        iso = _normalize_meeting_date(m.group(1))
        if iso:
            return iso

    for key in ("date", "날짜", "일자"):
        m2 = re.search(rf"(?:\*\*)?\s*{key}\s*(?:\*\*)?\s*:\s*([^\n\r]+)", transcript, flags=re.I)
        if m2:
            iso = _normalize_meeting_date(m2.group(1))
            if iso:
                return iso

    head = "\n".join(transcript.splitlines()[:20])
    guess = _normalize_meeting_date(head)
    return guess

def _normalize_actions(actions: Any, meeting_date_dt: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """
    actions 배열을 정규화:
    - 키 제한(owner, task, due)
    - 문자열 트림
    - due ISO 형식만 허용(아니면 None)
    - meeting_date가 주어지면 과거 날짜 due는 None 처리
    """
    normalized: List[Dict[str, Any]] = []
    if not isinstance(actions, list):
        return normalized

    for a in actions:
        owner = (a.get("owner") if isinstance(a, dict) else "") or ""
        task  = (a.get("task")  if isinstance(a, dict) else "") or ""
        due   = (a.get("due")   if isinstance(a, dict) else None)

        owner = owner.strip()
        task  = task.strip()

        if _is_valid_iso_date(due):
            try:
                due_dt = dt.date.fromisoformat(due)
            except ValueError:
                due_dt = None
        else:
            due_dt = None

        if due_dt and meeting_date_dt and due_dt < meeting_date_dt:
            due_dt = None

        due_val: Optional[str] = due_dt.isoformat() if due_dt else None
        normalized.append({"owner": owner, "task": task, "due": due_val})
    return normalized

def _apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    data.setdefault("brief", "")
    data.setdefault("bullets", [])
    data.setdefault("decisions", [])
    data.setdefault("actions", [])
    th = (data.get("type_hint") or "").strip().lower()
    if th not in ("research", "general"):
        th = "general"
    data["type_hint"] = th
    return data

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def summarize_transcript(
    transcript: str,
    api_key: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,  # ★ app.py의 기본데이터 전달
    model: str = "gpt-4o-mini",
    allow_transcript_date_fallback: bool = False,  # ★ 기본은 meta에서만 날짜 사용
) -> Dict[str, Any]:
    """
    전사본 -> 요약(JSON). 연구회의 추정 시 research_enrich 포함.

    meeting_date는 원칙적으로 app.py의 기본데이터(meta)에서 추출합니다.
    - meta['dt'|'date'|'meeting_date'|'일시'|'날짜'|'일자'] 중 첫 값을 ISO(YYYY-MM-DD)로 정규화
    - allow_transcript_date_fallback=True 인 경우에만, meta가 비어 있으면 전사 헤더에서 보조 추출
    """
    client = get_client(api_key)

    # 0) meeting_date: meta 우선, 기본적으로 transcript fallback 사용 안 함
    meeting_date_iso = _extract_meeting_date_from_meta(meta)
    if not meeting_date_iso and allow_transcript_date_fallback:
        meeting_date_iso = _extract_meeting_date_from_text(transcript)

    meeting_date_dt: Optional[dt.date] = None
    if meeting_date_iso:
        try:
            meeting_date_dt = dt.date.fromisoformat(meeting_date_iso)
        except ValueError:
            meeting_date_iso = None
            meeting_date_dt = None

    # 1) 프롬프트 구성
    transcript_snip = transcript[:18000] if transcript else ""
    mt_line = f"\nmeeting_date: {meeting_date_iso}\n" if meeting_date_iso else "\nmeeting_date: (미지정)\n"

    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": mt_line + SUMMARY_PROMPT.replace("{{transcript}}", transcript_snip),
        },
    ]

    # 2) 1차 요약
    try:
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = r.choices[0].message.content or "{}"
    except Exception:
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.2,
        )
        text = r.choices[0].message.content or "{}"

    data = _safe_json_loads(text)
    data = _apply_defaults(data)
    data["actions"] = _normalize_actions(data.get("actions"), meeting_date_dt)

    # 3) 연구 보조요약 (연구로 감지된 경우)
    if data["type_hint"] == "research":
        enrich_msgs = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    (f"meeting_date: {meeting_date_iso}\n" if meeting_date_iso else "meeting_date: (미지정)\n")
                    + RESEARCH_ENRICH
                    + "\n\n---\n[요약(JSON)]:\n"
                    + json.dumps(
                        {
                            "brief": data.get("brief", ""),
                            "bullets": data.get("bullets", []),
                            "decisions": data.get("decisions", []),
                            "actions": data.get("actions", []),
                            "type_hint": data.get("type_hint", "general"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n[전사 일부]:\n"
                    + transcript_snip
                ),
            },
        ]

        try:
            r2 = client.chat.completions.create(
                model=model,
                messages=enrich_msgs,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            enrich_text = r2.choices[0].message.content or "{}"
        except Exception:
            r2 = client.chat.completions.create(
                model=model,
                messages=enrich_msgs,
                temperature=0.2,
            )
            enrich_text = r2.choices[0].message.content or "{}"

        enrich_data = _safe_json_loads(enrich_text)

        enrich_data.setdefault("objective", "")
        enrich_data.setdefault("methods", [])
        enrich_data.setdefault("results", [])
        enrich_data.setdefault("limitations", "")
        enrich_data.setdefault("actions", [])
        enrich_data["actions"] = _normalize_actions(enrich_data.get("actions"), meeting_date_dt)

        data["research_enrich"] = enrich_data

    return data
