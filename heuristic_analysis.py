from __future__ import annotations

import re
from collections import defaultdict

VALUE_RE = re.compile(r"(?:₹|Rs\.?|INR)?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|x|times|crore|cr|lakh crore|bn|billion|mn|million|₹)?", re.I)
SENT_RE = re.compile(r"(?<=[.!?])\s+")

POSITIVE_TERMS = [
    "growth", "grew", "increase", "increased", "improved", "strong", "robust", "higher",
    "recovery", "expanded", "expansion", "profit", "margin", "demand", "outperformed",
    "record", "best-ever", "leadership", "opportunity", "target", "guidance"
]
CAUTION_TERMS = [
    "risk", "decline", "declined", "lower", "pressure", "compressed", "compression", "weak",
    "concern", "challenge", "unclear", "missing", "not available", "pending", "volatile",
    "cost", "debt", "loss", "penalty", "regulatory", "delay", "slowdown", "uncertain"
]

GENERIC_METRICS = [
    "revenue", "sales", "ebitda", "pat", "profit", "margin", "cash", "debt", "capex",
    "target", "guidance", "valuation", "price", "date", "timeline", "growth", "volume",
    "cost", "risk", "customer", "market", "approval", "trial", "policy", "compliance",
    "endpoint", "safety", "adverse", "financing", "implementation", "accountability"
]

RISK_BY_TERM = {
    "revenue": "Financial Data Risk", "sales": "Financial Data Risk", "ebitda": "Financial Data Risk",
    "pat": "Financial Data Risk", "profit": "Financial Data Risk", "margin": "Financial Data Risk",
    "cash": "Financial Data Risk", "debt": "Financial Data Risk", "capex": "Financial Data Risk",
    "target": "Time Period Risk", "guidance": "Time Period Risk", "valuation": "Recommendation Risk",
    "volume": "Operational Risk", "cost": "Operational Risk", "market": "Market / Commercial Risk",
    "approval": "Regulatory / Compliance Risk", "trial": "Regulatory / Compliance Risk", "endpoint": "Definition / Methodology Risk",
    "safety": "Regulatory / Compliance Risk", "adverse": "Regulatory / Compliance Risk", "policy": "Regulatory / Compliance Risk",
    "implementation": "Operational Risk", "financing": "Financial Data Risk", "accountability": "Regulatory / Compliance Risk",
}

def _sentences(text: str) -> list[str]:
    parts = SENT_RE.split(re.sub(r"\s+", " ", text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 40]

def _query_terms(query: str) -> list[str]:
    q = (query or "").lower()
    terms = [m for m in GENERIC_METRICS if m in q]
    if not terms:
        stop = {"are", "there", "any", "the", "and", "or", "in", "of", "for", "to", "what", "how", "is", "does", "do", "with", "a", "an"}
        terms = [w for w in re.findall(r"[a-zA-Z]{4,}", q) if w not in stop][:7]
    return terms[:10]

def _norm_value(v: str) -> str:
    raw = re.sub(r"\s+", " ", (v or "").strip().lower())
    num = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", raw)
    unit = "%" if "%" in raw else "crore" if ("crore" in raw or re.search(r"\bcr\b", raw)) else "x" if ("x" in raw or "times" in raw) else "number"
    if not num:
        return raw
    try:
        val = float(num.group(0).replace(",", ""))
        return f"{unit}:{round(val, 4)}"
    except Exception:
        return raw

def _risk_type(term: str) -> str:
    return RISK_BY_TERM.get((term or "").lower(), "Source Conflict Risk")

def _evidence_from_items(items, max_items=3):
    ev, seen = [], set()
    for c, s, _vals in items:
        key = (c.get("chunk_id"), s[:120])
        if key in seen:
            continue
        seen.add(key)
        ev.append({"chunk_id": c.get("chunk_id", "N/A"), "source": c.get("source", "Unknown"), "quote_or_phrase": s[:350]})
        if len(ev) >= max_items:
            break
    return ev

def heuristic_discrepancies(user_query: str, chunks: list[dict], limit: int = 3) -> list[dict]:
    terms = _query_terms(user_query)
    candidates = []
    by_term = defaultdict(list)
    for c in chunks:
        text = c.get("text", "")
        lower = text.lower()
        for term in terms:
            if term and term in lower:
                for s in _sentences(text):
                    if term in s.lower() and VALUE_RE.search(s):
                        vals = VALUE_RE.findall(s)
                        by_term[term].append((c, s, vals[:4]))
                        break

    for term, items in by_term.items():
        norm_vals = []
        for _, _, vals in items:
            for v in vals:
                nv = _norm_value(v)
                # drop bare years which create false conflicts
                if re.fullmatch(r"number:(19|20)\d{2}(\.0)?", nv):
                    continue
                if nv not in norm_vals:
                    norm_vals.append(nv)
        if len(items) >= 2 and len(norm_vals) >= 2:
            ev = _evidence_from_items(items, 3)
            candidates.append({
                "id": f"INC-{len(candidates)+1:03d}",
                "severity": "MEDIUM",
                "type": "Reconciliation Required / Potential Value Mismatch",
                "risk_type": _risk_type(term),
                "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
                "title": f"Potential mismatch around '{term}' requires reconciliation",
                "description": f"The retrieved chunks mention different values around '{term}'. This is a candidate issue, not automatically a confirmed contradiction, because the app must verify whether the period, scope, unit, and definition are comparable.",
                "evidence": ev,
                "assumptions_made": [
                    f"The term '{term}' is treated as relevant because it appears in the user query or retrieved chunks.",
                    "The values are treated as comparable only as an initial candidate; final confirmation requires checking period, scope, unit, and definition."
                ],
                "why_it_matters": "Unreconciled values can mislead the final answer if different scopes, time periods, or definitions are mixed together.",
                "suggested_resolution": "Check whether the values refer to the same period, source, scope, and unit. If not, label them separately instead of treating them as conflicting."
            })
        if len(candidates) >= limit:
            return candidates[:limit]

    unavailable, provided = [], []
    for c in chunks:
        for s in _sentences(c.get("text", "")):
            sl = s.lower()
            if any(x in sl for x in ["not available", "not disclosed", "missing", "unavailable"]):
                unavailable.append((c, s, []))
            elif VALUE_RE.search(s) and any(t in sl for t in terms):
                provided.append((c, s, VALUE_RE.findall(s)[:4]))
    if unavailable and provided:
        items = unavailable[:1] + provided[:2]
        ev = _evidence_from_items(items, 3)
        candidates.append({
            "id": f"INC-{len(candidates)+1:03d}",
            "severity": "HIGH",
            "type": "Source Conflict",
            "risk_type": "Missing Evidence Risk",
            "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
            "title": "Information appears unavailable in one place but available elsewhere",
            "description": "One retrieved chunk suggests information is unavailable or missing, while another retrieved chunk appears to provide related evidence. This needs reconciliation.",
            "evidence": ev,
            "assumptions_made": ["The unavailable statement and provided evidence are assumed to relate to the same user-requested topic until source context proves otherwise."],
            "why_it_matters": "The system should not tell the user that evidence is unavailable if another retrieved chunk contains relevant evidence.",
            "suggested_resolution": "Confirm whether the available evidence is the same metric/claim and whether it is current, comparable, and source-supported."
        })
    return candidates[:limit]

def heuristic_novel_insights(user_query: str, chunks: list[dict], limit: int = 3) -> list[dict]:
    terms = _query_terms(user_query)
    positives, cautions, valued = [], [], []
    for c in chunks:
        for s in _sentences(c.get("text", "")):
            sl = s.lower()
            relevant = any(t in sl for t in terms) or any(t in sl for t in POSITIVE_TERMS + CAUTION_TERMS)
            if not relevant:
                continue
            rec = (c, s, VALUE_RE.findall(s)[:4])
            if any(t in sl for t in POSITIVE_TERMS): positives.append(rec)
            if any(t in sl for t in CAUTION_TERMS): cautions.append(rec)
            if VALUE_RE.search(s): valued.append(rec)
    insights = []
    if positives and cautions:
        ev = _evidence_from_items(positives[:1] + cautions[:2], 3)
        insights.append({
            "id": "NOV-001",
            "strength": "HIGH",
            "category": "Evidence Tension",
            "risk_type": "Narrative vs Metric Risk",
            "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
            "title": "Positive narrative should be read alongside caution signals",
            "insight": "The retrieved chunks contain both positive performance/growth language and cautionary signals. The useful insight is not just the headline improvement, but whether the improvement is sustainable after considering the disclosed risks, pressures, or unclear assumptions.",
            "evidence": ev,
            "assumptions_made": ["Positive and cautionary statements are assumed to relate to the same broad user-requested topic; the reviewer should verify whether they share the same period and scope."],
            "why_it_matters": "This helps the user avoid accepting a positive headline without checking the conditions that could weaken it.",
            "suggested_follow_up": "Ask whether the positive trend is supported by recurring drivers, comparable metrics, and quantified risk impact."
        })
    if len(valued) >= 2:
        ev = _evidence_from_items(valued, 3)
        insights.append({
            "id": f"NOV-{len(insights)+1:03d}",
            "strength": "MEDIUM",
            "category": "Trend Pattern",
            "risk_type": "Definition / Methodology Risk",
            "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
            "title": "The answer may depend on connecting multiple metrics, not one fact",
            "insight": "Several retrieved chunks contain numeric evidence relevant to the query. A stronger answer should connect these values rather than summarize them separately, because the relationship between the metrics may reveal the real driver or risk.",
            "evidence": ev,
            "assumptions_made": ["The extracted numeric evidence is treated as analytically related because it was retrieved for the same query."],
            "why_it_matters": "Novelty often comes from the relationship between numbers, such as growth versus margin, profit versus cash, target versus actual, or commitment versus implementation.",
            "suggested_follow_up": "Ask for a metric/claim bridge using the same retrieved chunks."
        })
    return insights[:limit]
