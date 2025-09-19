def decide_doc_type(summary_json: dict) -> str:
    # prefer model hint; fallback to simple heuristics
    hint = (summary_json.get("type_hint") or "").lower()
    if hint in {"research"}:
        return "research"
    if hint in {"general","meeting","일반","회의"}:
        return "general"
    # heuristics based on bullets
    bullets = " ".join(summary_json.get("bullets", []))
    keys = ["실험", "IRB", "프로토콜", "피험자", "데이터셋", "분석계획", "hypothesis", "protocol", "assay"]
    if any(k in bullets for k in keys):
        return "research"
    return "general"
