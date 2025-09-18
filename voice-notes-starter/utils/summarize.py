# utils/summarize.py  (드롭인 교체)
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os, json, re

SYSTEM = "You are a Korean note-taking assistant for hospital meetings. Reply ONLY with valid JSON (UTF-8). No code fences, no prose."

SUMMARY_PROMPT = """
다음은 병원 회의의 전사본이야. 핵심을 한국어로 간결하게 정리해.
응답은 JSON으로만 하고, 키는
- brief: 2-3문장 요약
- bullets: 5개 내외 핵심 bullet 리스트
- decisions: 회의에서 결정된 사항 bullet
- actions: 액션 배열 (각 항목은 {owner, task, due})
- type_hint: 'research' 또는 'general' 중 하나 (가능성 판단)

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
"""

def get_client(api_key: str | None = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)

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
    raise json.JSONDecodeError("JSON parsing failed", s[:200] + ("..." if len(s) > 200 else ""), 0)

def summarize_transcript(transcript: str, api_key: str | None = None) -> Dict[str, Any]:
    client = get_client(api_key)

    # 1) 1차 요약 (JSON 강제)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SUMMARY_PROMPT.replace("{{transcript}}", transcript[:18000])},
    ]
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.2,
            response_format={"type": "json_object"},  # ✅ JSON 강제
        )
        text = r.choices[0].message.content or "{}"
    except Exception:
        # 오래된 SDK/모델에서 response_format 미지원일 가능성 폴백
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.2,
        )
        text = r.choices[0].message.content or "{}"

    data = _safe_json_loads(text)

    # 기본 키 보정
    data.setdefault("brief", "")
    data.setdefault("bullets", [])
    data.setdefault("decisions", [])
    data.setdefault("actions", [])
    type_hint = (data.get("type_hint") or "").lower()

    # 2) 연구 보조요약 (연구로 감지된 경우에만, JSON 강제 + 컨텍스트 제공)
    if type_hint.startswith("research"):
        enrich_msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                RESEARCH_ENRICH
                + "\n\n---\n[요약(JSON)]:\n"
                + json.dumps(data, ensure_ascii=False)
                + "\n\n[전사 일부]:\n"
                + transcript[:18000]
            },
        ]
        try:
            r2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=enrich_msgs,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            enrich_text = r2.choices[0].message.content or "{}"
        except Exception:
            r2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=enrich_msgs,
                temperature=0.2,
            )
            enrich_text = r2.choices[0].message.content or "{}"

        data["research_enrich"] = _safe_json_loads(enrich_text)

        # 보조요약에서 필요한 키가 비어있어도 템플릿이 작동하도록 기본값 보정
        re_en = data["research_enrich"]
        re_en.setdefault("objective", "")
        re_en.setdefault("methods", [])
        re_en.setdefault("results", [])
        re_en.setdefault("limitations", "")
        re_en.setdefault("actions", [])

    return data
