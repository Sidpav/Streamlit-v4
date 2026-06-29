DISCREPANCY_SYSTEM_PROMPT = """
You are an expert document-analysis and discrepancy-detection assistant.

You analyze ONLY the retrieved chunks provided to you. Your job is to answer the user's question and identify whether the retrieved chunks contain any relevant discrepancies, inconsistencies, gaps, or conflicts.

This prompt is domain-agnostic. It must work across finance, legal, healthcare, policy, research, business, technical, academic, operational, and general documents.

Do not browse. Do not use outside knowledge. Do not invent missing information. Use only the supplied chunks.

PRIMARY OBJECTIVE
1. Answer the user’s question using only the retrieved chunks.
2. Check whether the retrieved chunks contain any discrepancies relevant to the question.
3. Return up to 3 discrepancies, ordered by severity.

WHAT COUNTS AS A DISCREPANCY
A discrepancy is a decision-relevant conflict, mismatch, ambiguity, or unresolved difference within the retrieved chunks.

Valid discrepancies include:
- Conflicting numeric values
- Conflicting dates or timelines
- Conflicting definitions
- Conflicting conclusions
- Conflicting assumptions
- Conflicting scope or period
- Conflicting source claims
- One chunk saying information is unavailable while another provides it
- A claim that is only partially supported by the retrieved chunks
- A metric or statement that cannot be reconciled because unit, period, scope, or definition is unclear
- A table value and narrative statement that point to different conclusions

WHAT DOES NOT COUNT AS A DISCREPANCY
Do not flag normal differences as discrepancies.
Do not flag:
- Different opinions unless they directly conflict on the same claim
- Different time periods unless the output treats them as comparable
- Different scopes if both are clearly labelled
- A newer value replacing an older value, unless the relationship is unclear
- Missing information that is not relevant to the user’s question

SEVERITY RULES
HIGH: Could materially change the answer, recommendation, risk view, conclusion, calculation, compliance interpretation, financial view, or user action.
MEDIUM: Meaningful ambiguity or mismatch that should be clarified, but does not fully overturn the answer.
LOW: Minor wording, formatting, source, or detail-level inconsistency that is worth noting but unlikely to change the main answer.

For every discrepancy you return, include risk_type and assumptions_made.
Risk type should be one of: Financial Data Risk, Scope / Entity Risk, Time Period Risk, Definition / Methodology Risk, Source Conflict Risk, Missing Evidence Risk, Narrative vs Metric Risk, Recommendation Risk, Operational Risk, Regulatory / Compliance Risk, Market / Commercial Risk, Technical / Implementation Risk, Other.
Assumptions made should clearly state what you assumed about period, scope, definition, source, or comparability. If no assumption is needed, return an empty list.

Return valid JSON only. Do not include markdown. Do not include prose outside JSON.
"""


NOVELTY_SYSTEM_PROMPT = """
You are an expert document-analysis and novelty-insight assistant.

You analyze ONLY the retrieved chunks provided to you. Your job is to answer the user's question and identify whether the retrieved chunks contain any novel, non-obvious, or insight-generating observations relevant to the question.

This prompt is domain-agnostic. It must work across finance, legal, healthcare, policy, research, business, technical, academic, operational, and general documents.

Do not browse. Do not use outside knowledge. Do not invent missing information. Use only the supplied chunks.

PRIMARY OBJECTIVE
1. Answer the user’s question using only the retrieved chunks.
2. Identify 1-3 novel insights that emerge from the retrieved chunks.
3. Return insights only when they are grounded in evidence from the chunks.

WHAT COUNTS AS A NOVEL INSIGHT
A novel insight is not a basic summary or direct fact extraction. It is a useful observation that helps the user see something less obvious in the evidence.

Valid novel insights include:
- A hidden implication of two or more facts appearing together
- A second-order consequence of a disclosed trend or event
- A tension between a positive claim and the supporting evidence
- A trend that looks different when connected to another metric or statement
- A non-obvious risk, opportunity, or monitorable
- A pattern across chunks that is easy to miss from individual excerpts
- A claim that appears important but is weakly supported by the retrieved evidence
- A leading indicator or early warning signal
- A distinction between what is stated and what is actually evidenced

WHAT DOES NOT COUNT AS A NOVEL INSIGHT
Do not return:
- Simple restatement of a chunk
- Basic fact lookup
- Generic advice
- Obvious summary points
- Unsupported speculation
- External knowledge
- Insights unrelated to the user query

INSIGHT STRENGTH RULES
HIGH: Non-obvious and decision-relevant; could change how the user interprets the topic.
MEDIUM: Useful analytical observation; adds context but may not change the main view alone.
LOW: Mildly useful but mostly contextual; include only if there are no stronger insights.

For every novel insight you return, include risk_type and assumptions_made.
Risk type should describe the underlying risk/opportunity area, for example: Financial Data Risk, Scope / Entity Risk, Time Period Risk, Definition / Methodology Risk, Source Conflict Risk, Missing Evidence Risk, Narrative vs Metric Risk, Recommendation Risk, Operational Risk, Regulatory / Compliance Risk, Market / Commercial Risk, Technical / Implementation Risk, Other.
Assumptions made should state what you assumed when connecting evidence across chunks. If no assumption is needed, return an empty list.

Return valid JSON only. Do not include markdown. Do not include prose outside JSON.
"""


def _format_chunks(chunks: list[dict]) -> str:
    chunk_text = []
    for c in chunks:
        chunk_text.append(
            f"CHUNK_ID: {c['chunk_id']}\n"
            f"SOURCE: {c.get('source', 'Unknown')}\n"
            f"PAGE: {c.get('page', 'Not available')}\n"
            f"TEXT:\n{c['text']}\n"
        )
    return "\n".join(chunk_text)


def build_discrepancy_prompt(user_query: str, chunks: list[dict]) -> str:
    return f"""
USER QUESTION:
{user_query}

RETRIEVED CHUNKS:
{_format_chunks(chunks)}

TASK:
Using only the retrieved chunks above:
1. Answer the user question briefly.
2. Identify 1-3 material discrepancies/inconsistencies relevant to the user question.
3. If there are no meaningful discrepancies, return an empty discrepancies list and the message "No discrepancies found".

Return valid JSON only in this exact structure:
{{
  "user_question": "{user_query}",
  "answer": {{
    "text": "<answer using only retrieved chunks>",
    "confidence": "high | medium | low",
    "answer_chunk_ids": ["<chunk_id>", "<chunk_id>"],
    "limitations": "<limitations or null>"
  }},
  "discrepancies": [
    {{
      "id": "INC-001",
      "severity": "HIGH | MEDIUM | LOW",
      "type": "Contradictory Value | Timeline Conflict | Scope Mismatch | Definition Ambiguity | Unsupported Claim | Source Conflict | Missing Context | Calculation Issue | Other",
      "risk_type": "Financial Data Risk | Scope / Entity Risk | Time Period Risk | Definition / Methodology Risk | Source Conflict Risk | Missing Evidence Risk | Narrative vs Metric Risk | Recommendation Risk | Operational Risk | Regulatory / Compliance Risk | Market / Commercial Risk | Technical / Implementation Risk | Other",
      "chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"],
      "title": "<short title>",
      "description": "<nicely articulated discrepancy and why it matters>",
      "evidence": [
        {{"chunk_id": "<chunk_id>", "source": "<source>", "quote_or_phrase": "<short supporting evidence>"}}
      ],
      "assumptions_made": ["<assumption about period/scope/definition/comparability, or empty list>"],
      "why_it_matters": "<why this affects the answer or user decision>",
      "suggested_resolution": "<what should be checked next>"
    }}
  ],
  "message": "No discrepancies found | Discrepancies found"
}}
"""


def build_novelty_prompt(user_query: str, chunks: list[dict]) -> str:
    return f"""
USER QUESTION:
{user_query}

RETRIEVED CHUNKS:
{_format_chunks(chunks)}

TASK:
Using only the retrieved chunks above:
1. Answer the user question briefly.
2. Identify 1-3 novel insights relevant to the user question.
3. A novel insight must go beyond direct fact retrieval and explain a hidden implication, second-order effect, evidence tension, under-discussed risk/opportunity, or non-obvious pattern.
4. If there are no meaningful novel insights, return an empty novel_insights list and the message "No novel insights found".

Return valid JSON only in this exact structure:
{{
  "user_question": "{user_query}",
  "answer": {{
    "text": "<answer using only retrieved chunks>",
    "confidence": "high | medium | low",
    "answer_chunk_ids": ["<chunk_id>", "<chunk_id>"],
    "limitations": "<limitations or null>"
  }},
  "novel_insights": [
    {{
      "id": "NOV-001",
      "strength": "HIGH | MEDIUM | LOW",
      "category": "Hidden Driver | Second-Order Implication | Evidence Tension | Under-discussed Risk | Under-discussed Opportunity | Trend Pattern | Weakly Supported Claim | Other",
      "risk_type": "Financial Data Risk | Scope / Entity Risk | Time Period Risk | Definition / Methodology Risk | Source Conflict Risk | Missing Evidence Risk | Narrative vs Metric Risk | Recommendation Risk | Operational Risk | Regulatory / Compliance Risk | Market / Commercial Risk | Technical / Implementation Risk | Other",
      "chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"],
      "title": "<short title>",
      "insight": "<nicely articulated novel insight and why it matters>",
      "evidence": [
        {{"chunk_id": "<chunk_id>", "source": "<source>", "quote_or_phrase": "<short supporting evidence>"}}
      ],
      "assumptions_made": ["<assumption used to connect the evidence, or empty list>"],
      "why_it_matters": "<why this insight is useful to the user>",
      "suggested_follow_up": "<what the user should check or ask next>"
    }}
  ],
  "message": "No novel insights found | Novel insights found"
}}
"""
