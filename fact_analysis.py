from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

SENT_RE = re.compile(r"(?<=[.!?])\s+")
VALUE_RE = re.compile(
    r"(?:₹|Rs\.?|INR)?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|x|times|crore|cr|lakh crore|bn|billion|mn|million|MT|mt|million tonnes|kt|KT)?",
    re.I,
)
PERIOD_RE = re.compile(r"\b(?:FY\s*\d{2,4}(?:[-–]\d{2,4})?|Q[1-4]\s*(?:FY)?\s*\d{2,4}|full year|year[- ]end(?:ed)?|quarter|9M|H1|H2)\b", re.I)

METRIC_PATTERNS: Dict[str, List[str]] = {
    "Revenue / Sales": ["revenue", "revenues", "sales", "turnover", "income from operations"],
    "EBITDA": ["ebitda", "operating profit"],
    "EBITDA Margin": ["ebitda margin", "operating margin", "margin"],
    "PAT / Net Profit": ["pat", "profit after tax", "net profit", "profit for the year", "reported pat"],
    "Net Debt": ["net debt", "debt", "borrowings", "leverage"],
    "Capex": ["capex", "capital expenditure", "capital investment"],
    "Cash Flow": ["free cash flow", "operating cash flow", "cash flow", "cash generation", "fcf"],
    "Volume / Deliveries": ["volume", "volumes", "deliveries", "production", "crude steel", "shipments"],
    "Guidance / Target": ["guidance", "target", "expects", "expected", "outlook", "forecast"],
    "Margin / Cost Pressure": ["margin", "cost", "pressure", "discounting", "logistics", "compression"],
}

SCOPE_PATTERNS: Dict[str, List[str]] = {
    "consolidated": ["consolidated", "group", "company level"],
    "standalone": ["standalone", "parent"],
    "india": ["india", "indian", "tata steel india"],
    "europe": ["europe", "european"],
    "uk": ["uk", "united kingdom", "britain"],
    "netherlands": ["netherlands", "nederland", "ijmuiden"],
    "segment": ["segment", "business segment", "division"],
    "retail": ["retail"],
    "jio": ["jio"],
    "o2c": ["o2c", "oil to chemicals"],
}

POSITIVE_TERMS = [
    "growth", "grew", "increase", "increased", "improved", "strong", "record", "higher", "recovery",
    "expanded", "expansion", "resilient", "robust", "profitability improved", "margin expansion", "optimistic"
]
CAUTION_TERMS = [
    "decline", "declined", "lower", "pressure", "compressed", "compression", "negative", "loss", "weak",
    "risk", "challenge", "debt", "capex", "cash flow", "cash generation", "uncertain", "volatile", "delay", "impairment"
]


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = SENT_RE.split(text)
    # also split long table-like chunks at semicolons/newline artifacts when needed
    out = []
    for p in parts:
        if len(p) > 700:
            out.extend([x.strip() for x in re.split(r";\s+|\s{2,}", p) if len(x.strip()) > 40])
        else:
            out.append(p.strip())
    return [p for p in out if len(p) > 12]


def _find_metric(sentence: str) -> str | None:
    sl = sentence.lower()
    # more specific first
    ordered = sorted(METRIC_PATTERNS.items(), key=lambda kv: max(len(x) for x in kv[1]), reverse=True)
    for metric, pats in ordered:
        for p in pats:
            if p in sl:
                return metric
    return None


def _find_period(sentence: str) -> str:
    m = PERIOD_RE.search(sentence)
    if m:
        return re.sub(r"\s+", " ", m.group(0).upper().replace(" ", ""))
    return "Unknown period"


def _find_scope(sentence: str, source: str = "") -> str:
    text = f"{sentence} {source}".lower()
    hits = []
    for scope, pats in SCOPE_PATTERNS.items():
        if any(p in text for p in pats):
            hits.append(scope)
    # Avoid classifying everything as Europe if a more specific geo is present
    if "netherlands" in hits:
        return "netherlands"
    if "uk" in hits:
        return "uk"
    if "india" in hits:
        return "india"
    if "consolidated" in hits:
        return "consolidated"
    if "standalone" in hits:
        return "standalone"
    return hits[0] if hits else "Unknown scope"


def _unit(raw_value: str) -> str:
    v = raw_value.lower()
    if "%" in v:
        return "%"
    if "lakh crore" in v:
        return "lakh crore"
    if "crore" in v or re.search(r"\bcr\b", v):
        return "crore"
    if "mt" in v or "million tonnes" in v:
        return "MT"
    if "bn" in v or "billion" in v:
        return "billion"
    if "mn" in v or "million" in v:
        return "million"
    if "x" in v or "times" in v:
        return "x"
    return "number"


def _numeric(raw_value: str) -> float | None:
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", raw_value)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None



def _normalized_value_key(raw_value: str) -> str:
    """Normalizes numeric values so identical values are not falsely flagged."""
    num = _numeric(raw_value)
    unit = _unit(raw_value)
    if num is None:
        return re.sub(r"\s+", " ", (raw_value or "").strip().lower())
    return f"{unit}:{round(num, 4)}"


def _risk_type_for_metric(metric: str, issue_type: str = "") -> str:
    metric = (metric or "").lower()
    issue_type = (issue_type or "").lower()
    if "scope" in issue_type:
        return "Scope / Entity Risk"
    if "period" in issue_type or "guidance" in issue_type or "target" in issue_type:
        return "Time Period Risk"
    if any(x in metric for x in ["revenue", "sales", "ebitda", "pat", "profit", "debt", "capex", "cash", "margin"]):
        return "Financial Data Risk"
    if any(x in metric for x in ["volume", "deliveries", "production"]):
        return "Operational Risk"
    return "Definition / Methodology Risk"


def _assumptions_for(metric: str, period: str, scope: str, issue_type: str) -> list[str]:
    assumptions = []
    if period == "Unknown period":
        assumptions.append("The period is not clearly detected from the retrieved sentence; this should be verified against the original source.")
    else:
        assumptions.append(f"The compared evidence is assumed to refer to {period} based on the retrieved text.")
    if scope == "Unknown scope":
        assumptions.append("The reporting scope/entity is not clearly detected; values may refer to consolidated, standalone, segment, geography, or another basis.")
    else:
        assumptions.append(f"The compared evidence is assumed to refer to the {scope} scope/entity based on detected wording/source context.")
    if "Value" in issue_type or "Conflict" in issue_type:
        assumptions.append("Values are treated as comparable only if metric, period, unit, and scope/entity are confirmed to match.")
    return assumptions


def _evidence_by_distinct_value(facts: List[dict], max_items: int = 4) -> List[dict]:
    ev = []
    seen_values = set()
    seen_rows = set()
    for f in facts:
        value_key = f.get("normalized_value_key") or _normalized_value_key(f.get("value", ""))
        row_key = (f.get("chunk_id"), f.get("sentence")[:100])
        if value_key in seen_values and len(seen_values) >= 2:
            continue
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        seen_values.add(value_key)
        ev.append({
            "chunk_id": f.get("chunk_id", "N/A"),
            "source": f.get("source", "Unknown"),
            "metric": f.get("metric", ""),
            "period": f.get("period", ""),
            "scope_entity": f.get("scope_entity", ""),
            "value": f.get("value", ""),
            "quote_or_phrase": f.get("sentence", "")[:350],
        })
        if len(ev) >= max_items:
            break
    return ev

def extract_structured_facts(chunks: List[dict], limit: int = 250) -> List[dict]:
    facts = []
    seen = set()
    for c in chunks:
        for s in _sentences(c.get("text", "")):
            metric = _find_metric(s)
            if not metric:
                continue
            values = VALUE_RE.findall(s)
            if not values:
                continue
            period = _find_period(s)
            scope = _find_scope(s, c.get("source", ""))
            for raw in values[:4]:
                raw = re.sub(r"\s+", " ", raw.strip())
                # Avoid treating the year inside FY24/FY2024 as a financial metric value.
                if re.fullmatch(r"\d{2,4}", raw) and re.search(rf"FY\s*{re.escape(raw)}", s, re.I):
                    continue
                key = (c.get("chunk_id"), metric, raw, s[:120])
                if key in seen:
                    continue
                seen.add(key)
                facts.append({
                    "metric": metric,
                    "value": raw,
                    "numeric_value": _numeric(raw),
                    "unit": _unit(raw),
                    "normalized_value_key": _normalized_value_key(raw),
                    "period": period,
                    "scope_entity": scope,
                    "chunk_id": c.get("chunk_id", "N/A"),
                    "source": c.get("source", "Unknown"),
                    "page": c.get("page", ""),
                    "sentence": s[:500],
                })
                if len(facts) >= limit:
                    return facts
    return facts


def _distinct_values(facts: List[dict]) -> List[str]:
    vals = []
    for f in facts:
        v = f.get("normalized_value_key") or _normalized_value_key(f.get("value", ""))
        if v and v not in vals:
            vals.append(v)
    return vals


def _evidence(facts: List[dict], max_items: int = 4) -> List[dict]:
    ev = []
    seen = set()
    for f in facts:
        key = (f.get("chunk_id"), f.get("sentence"))
        if key in seen:
            continue
        seen.add(key)
        ev.append({
            "chunk_id": f.get("chunk_id", "N/A"),
            "source": f.get("source", "Unknown"),
            "metric": f.get("metric", ""),
            "period": f.get("period", ""),
            "scope_entity": f.get("scope_entity", ""),
            "value": f.get("value", ""),
            "quote_or_phrase": f.get("sentence", "")[:350],
        })
        if len(ev) >= max_items:
            break
    return ev


def structured_discrepancies(facts: List[dict], limit: int = 5) -> List[dict]:
    findings = []

    # 1. Same metric + period + scope with different values = strongest.
    grouped = defaultdict(list)
    for f in facts:
        if f.get("metric") and f.get("value"):
            grouped[(f["metric"], f.get("period", "Unknown period"), f.get("scope_entity", "Unknown scope"), f.get("unit", ""))].append(f)

    for (metric, period, scope, unit), fs in grouped.items():
        vals = _distinct_values(fs)
        if len(fs) >= 2 and len(vals) >= 2 and metric not in {"Guidance / Target"}:
            severity = "HIGH" if period != "Unknown period" and scope != "Unknown scope" else "MEDIUM"
            findings.append({
                "id": f"INC-{len(findings)+1:03d}",
                "severity": severity,
                "type": "Confirmed Value Conflict" if severity == "HIGH" else "Potential Value Conflict",
                "risk_type": _risk_type_for_metric(metric, "Value Conflict"),
                "chunk_ids": list(dict.fromkeys([f["chunk_id"] for f in fs])),
                "title": f"{metric} values differ for {period}, {scope}",
                "description": f"The retrieved chunks contain different {metric} values with the same detected period ({period}) and scope/entity ({scope}). This is a stronger candidate inconsistency than a generic repeated metric because the comparison keys match.",
                "evidence": _evidence_by_distinct_value(fs),
                "assumptions_made": _assumptions_for(metric, period, scope, "Value Conflict"),
                "why_it_matters": f"Using the wrong {metric} value can change the financial interpretation, valuation, or recommendation. The analyst should not combine these figures unless the definition and source are reconciled.",
                "suggested_resolution": "Verify the original source tables and confirm whether the values are reported, adjusted, consolidated, standalone, segment-level, quarterly, or full-year figures.",
            })
            if len(findings) >= limit:
                return findings[:limit]

    # 2. Same metric + period but multiple scopes = reconciliation issue, not contradiction.
    grouped2 = defaultdict(list)
    for f in facts:
        grouped2[(f["metric"], f.get("period", "Unknown period"), f.get("unit", ""))].append(f)

    for (metric, period, unit), fs in grouped2.items():
        scopes = sorted(set(f.get("scope_entity", "Unknown scope") for f in fs))
        vals = _distinct_values(fs)
        if len(fs) >= 3 and len(vals) >= 2 and len(scopes) >= 2 and metric not in {"Guidance / Target"}:
            findings.append({
                "id": f"INC-{len(findings)+1:03d}",
                "severity": "MEDIUM",
                "type": "Reconciliation Required / Scope Mismatch",
                "risk_type": _risk_type_for_metric(metric, "Scope Mismatch"),
                "chunk_ids": list(dict.fromkeys([f["chunk_id"] for f in fs])),
                "title": f"{metric} needs scope separation for {period}",
                "description": f"The retrieved chunks mention different {metric} values for {period}, but the detected scopes/entities differ: {', '.join(scopes)}. This may not be a true contradiction; it is a reconciliation issue that should be separated before final analysis.",
                "evidence": _evidence_by_distinct_value(fs),
                "assumptions_made": _assumptions_for(metric, period, "multiple scopes", "Scope Mismatch"),
                "why_it_matters": "A report can become misleading if consolidated, standalone, geography-level, segment-level, or source-specific values are mixed together as if they are comparable.",
                "suggested_resolution": "Create separate rows for each scope/entity and only compare figures that share the same metric, period, unit, and scope.",
            })
            if len(findings) >= limit:
                return findings[:limit]

    # 3. Guidance divergence.
    guidance = [f for f in facts if f["metric"] == "Guidance / Target"]
    if len(guidance) >= 2 and len(_distinct_values(guidance)) >= 2:
        findings.append({
            "id": f"INC-{len(findings)+1:03d}",
            "severity": "MEDIUM",
            "type": "Guidance / Target Mismatch",
            "risk_type": "Time Period Risk",
            "chunk_ids": list(dict.fromkeys([f["chunk_id"] for f in guidance])),
            "title": "Guidance or target figures differ across retrieved chunks",
            "description": "The retrieved chunks contain multiple guidance/target values. This may reflect different scenarios, periods, or assumptions, but the difference should be made explicit before using the guidance in an answer.",
            "evidence": _evidence_by_distinct_value(guidance),
            "assumptions_made": ["Different guidance/target values are assumed to be comparable only if they refer to the same metric, period, scenario, and source basis."],
            "why_it_matters": "Guidance affects forward-looking conclusions and valuation. Mixing base-case and growth-case assumptions can overstate or understate the outlook.",
            "suggested_resolution": "Separate guidance by period, scenario, source, and metric. Label base case, growth case, management target, and analyst estimate distinctly.",
        })

    return findings[:limit]


def structured_novel_insights(facts: List[dict], chunks: List[dict], limit: int = 5) -> List[dict]:
    insights = []
    all_text = " ".join(c.get("text", "") for c in chunks)
    low = all_text.lower()

    def chunk_evidence(include_terms: List[str], max_items: int = 4) -> List[dict]:
        ev = []
        seen = set()
        for c in chunks:
            for s in _sentences(c.get("text", "")):
                sl = s.lower()
                if any(t in sl for t in include_terms):
                    key = (c.get("chunk_id"), s[:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    ev.append({"chunk_id": c.get("chunk_id", "N/A"), "source": c.get("source", "Unknown"), "quote_or_phrase": s[:350]})
                    break
            if len(ev) >= max_items:
                break
        return ev

    # India strength vs Europe/UK/Netherlands drag pattern.
    if any(t in low for t in ["india", "indian"]) and any(t in low for t in ["europe", "netherlands", "nederland", "uk"]):
        if any(t in low for t in ["negative ebitda", "loss", "pressure", "decline", "weak"]):
            ev = chunk_evidence(["india", "netherlands", "europe", "uk", "negative ebitda", "loss", "pressure"])
            insights.append({
                "id": f"NOV-{len(insights)+1:03d}",
                "strength": "HIGH",
                "category": "Segment / Geography Mix",
                "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
                "risk_type": "Scope / Entity Risk",
                "title": "Headline performance may hide segment/geography divergence",
                "insight": "The retrieved chunks suggest that different parts of the business may be moving in different directions. A consolidated view could hide strength in one geography or segment and weakness in another, so the most useful analysis is a split view rather than a single group-level conclusion.",
                "evidence": ev,
                "assumptions_made": ["Geography/segment-level statements are assumed to be analytically related because they were retrieved for the same user query; confirm matching period before final use."],
                "why_it_matters": "For investment analysis, segment/geography divergence can change the quality of earnings, valuation multiple, and risk view.",
                "suggested_follow_up": "Ask for a table separating consolidated, standalone, India, Europe, UK, and Netherlands metrics for the same period.",
            })

    # Growth / profit vs cash/debt/capex tension.
    if any(t in low for t in ["growth", "revenue", "profit", "ebitda", "pat"]) and any(t in low for t in ["debt", "capex", "cash flow", "free cash flow", "inventory", "working capital"]):
        ev = chunk_evidence(["growth", "revenue", "ebitda", "pat", "debt", "capex", "cash flow", "working capital", "inventory"])
        insights.append({
            "id": f"NOV-{len(insights)+1:03d}",
            "strength": "HIGH",
            "category": "Quality of Growth / Cash Conversion",
            "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
            "risk_type": "Financial Data Risk",
                "title": "Growth needs to be tested against cash conversion and balance sheet pressure",
            "insight": "The important analytical question is not only whether revenue, EBITDA, or PAT improved. The retrieved evidence should be connected to debt, capex, and cash-flow indicators to judge whether the growth is self-funding or capital-intensive.",
            "evidence": ev,
            "why_it_matters": "A company can show healthy headline growth while still creating weaker free cash flow or higher leverage. This changes the investment conclusion.",
            "suggested_follow_up": "Ask for a bridge from EBITDA/PAT to operating cash flow, free cash flow, capex, and net debt.",
        })

    # Margin improvement vs cost/pressure tension.
    if any(t in low for t in ["margin", "ebitda margin"]) and any(t in low for t in ["cost", "pressure", "discount", "logistics", "compression", "negative"]):
        ev = chunk_evidence(["margin", "ebitda margin", "cost", "pressure", "discount", "logistics", "compression", "negative"])
        insights.append({
            "id": f"NOV-{len(insights)+1:03d}",
            "strength": "MEDIUM",
            "category": "Evidence Tension",
            "chunk_ids": list(dict.fromkeys([e["chunk_id"] for e in ev])),
            "risk_type": "Narrative vs Metric Risk",
                "title": "Margin narrative needs a driver-level check",
            "insight": "If margins are discussed alongside cost pressure or weak segment performance, the novel insight is to separate structural margin improvement from temporary or mix-driven effects.",
            "evidence": ev,
            "why_it_matters": "Margin quality affects forward EBITDA assumptions and valuation confidence.",
            "suggested_follow_up": "Ask which margin drivers are recurring versus one-off, geography-specific, or segment-specific.",
        })

    return insights[:limit]
