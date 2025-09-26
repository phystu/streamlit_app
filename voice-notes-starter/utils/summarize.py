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
    """
    모델이 실수로 텍스트를 섞었을 때를 대비해 첫 번째 JSON blob만 추출/파싱.
    """
    # 빠른 경로
    try:
        return json.loads(s)
    except Exception:
        pass
    # 넓게 { ... } 블록만 추출
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        return json.loads(m.group(0))
    # 진짜로 JSON이 아니면 원문을 보여주기 쉬운 형태로 예외
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

def _normalize_actions(actions: Any) -> List[Dict[str, Any]]:
    """
    actions 배열을 정규화: 키 제한, 문자열 트림, due ISO 형식만 허용(아니면 None).
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
            due_val: Optional[str] = due
        else:
            due_val = None

        normalized.append({"owner": owner, "task": task, "due": due_val})
    return normalized

def _apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    data.setdefault("brief", "")
    data.setdefault("bullets", [])
    data.setdefault("decisions", [])
    data.setdefault("actions", [])
    # type_hint은 소문자 보정
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
    meeting_date: Optional[str] = None,  # 'YYYY-MM-DD' (KST 기준) 권장
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    전사본 -> 요약(JSON). 연구회의 추정 시 research_enrich 포함.
    - meeting_date: 상대기한 해석 기준일(선택). 'YYYY-MM-DD'. 없으면 due는 null로 유도.
    - model: 교체 용이하도록 인자로 노출(기본 gpt-4o-mini).
    """
    client = get_client(api_key)

    # 프롬프트 주입값 구성
    transcript_snip = transcript[:18000] if transcript else ""
    mt_line = f"\nmeeting_date: {meeting_date}\n" if meeting_date else "\nmeeting_date: (미지정)\n"

    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": mt_line + SUMMARY_PROMPT.replace("{{transcript}}", transcript_snip),
        },
    ]

    # 1) 1차 요약 (JSON 강제 시도)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = r.choices[0].message.content or "{}"
    except Exception:
        # 오래된 SDK/모델에서 response_format 미지원일 가능성 폴백
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.2,
        )
        text = r.choices[0].message.content or "{}"

    data = _safe_json_loads(text)
    data = _apply_defaults(data)
    data["actions"] = _normalize_actions(data.get("actions"))

    # 2) 연구 보조요약 (연구로 감지된 경우에만)
    if data["type_hint"] == "research":
        enrich_msgs = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    (f"meeting_date: {meeting_date}\n" if meeting_date else "meeting_date: (미지정)\n")
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

        # 보조요약 기본값 + 액션 정규화
        enrich_data.setdefault("objective", "")
        enrich_data.setdefault("methods", [])
        enrich_data.setdefault("results", [])
        enrich_data.setdefault("limitations", "")
        enrich_data.setdefault("actions", [])
        enrich_data["actions"] = _normalize_actions(enrich_data.get("actions"))

        data["research_enrich"] = enrich_data

    return data
