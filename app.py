from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from document_processing import chunk_documents, read_uploaded_file
from llm import QwenClient, normalize_discrepancy_payload, normalize_novelty_payload, parse_json_response
from heuristic_analysis import heuristic_discrepancies, heuristic_novel_insights
from fact_analysis import extract_structured_facts, structured_discrepancies, structured_novel_insights
from prompts import (
    DISCREPANCY_SYSTEM_PROMPT,
    NOVELTY_SYSTEM_PROMPT,
    build_discrepancy_prompt,
    build_novelty_prompt,
)
from retrieval import ChunkRetriever


st.set_page_config(page_title="Cortexa Chunk Intelligence Demo", layout="wide")

# Allow Streamlit Cloud secrets to behave like env vars for the demo.
for key in ["LLM_BACKEND", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "OLLAMA_MODEL", "QWEN_MODEL_ID"]:
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

st.title("Cortexa Chunk Intelligence Demo")
st.caption("Upload documents, ask a query, retrieve chunks, and run inconsistency + novelty analysis in one window. Version: v5 risk + assumptions + validation demo.")

with st.sidebar:
    st.header("Demo Settings")
    top_k = st.slider("Chunks to retrieve", min_value=5, max_value=60, value=30, step=5)
    experiment_counts = st.multiselect(
        "Experiment with chunk counts",
        options=[10, 15, 20, 30, 40, 50, 60],
        default=[10, 20, 30, 40],
        help="Run the same query with different retrieval counts to test when retrieved chunks become sufficient.",
    )
    chunk_words = st.slider("Chunk size", min_value=120, max_value=700, value=320, step=20)
    overlap_words = st.slider("Chunk overlap", min_value=0, max_value=200, value=60, step=10)
    st.divider()
    st.markdown("**Analysis Modules**")
    run_discrepancy_module = st.checkbox("Find gaps / inconsistencies", value=True)
    run_novelty_module = st.checkbox("Find novel insights", value=True)
    use_heuristic_fallback = st.checkbox(
        "Use demo fallback / rule-based backup if model returns empty output",
        value=True,
        help="Useful for demos and small models. If the LLM returns no inconsistencies/insights, the app surfaces evidence-grounded candidate findings from the retrieved chunks."
    )
    use_structured_analysis = st.checkbox(
        "Use structured financial fact extraction",
        value=True,
        help="Version 3-style upgrade: extracts metric, value, period, scope/entity, unit and compares only more comparable facts. Strongly recommended for financial documents."
    )
    st.divider()
    st.markdown("**Model Backend**")
    st.write(f"Current backend: `{os.getenv('LLM_BACKEND', 'mock')}`")
    st.caption("Use mock for UI testing. Use OpenRouter/Groq/Ollama/Transformers for real reasoning.")


st.warning(
    "For demo testing: keep fallback ON, use Top K 30 or 40, and upload the large Alpha Retail test pack. "
    "If this version is deployed correctly, the header should say 'Version: v5 risk + assumptions + validation demo'."
)

uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "docx", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_files:
    with st.spinner("Reading and chunking documents..."):
        raw_docs = []
        for file in uploaded_files:
            raw_docs.extend(read_uploaded_file(file))
        chunks = chunk_documents(raw_docs, chunk_words=chunk_words, overlap_words=overlap_words)
        st.session_state.chunks = chunks
        st.session_state.retriever = ChunkRetriever(chunks) if chunks else None

    st.success(f"Loaded {len(uploaded_files)} file(s), created {len(chunks)} chunks.")

    with st.expander("Preview chunks"):
        preview = pd.DataFrame([
            {"chunk_id": c.chunk_id, "source": c.source, "page": c.page, "text_preview": c.text[:240] + "..."}
            for c in chunks[:30]
        ])
        st.dataframe(preview, use_container_width=True)
else:
    st.info("Upload documents to begin.")

st.divider()
query = st.text_area(
    "User query",
    placeholder="Example: Are there any inconsistencies or novel insights in revenue, margins, cash generation, valuation, or risks?",
)
run = st.button("Run analysis", type="primary", disabled=not uploaded_files or not query.strip())


def run_one_query(user_query: str, k: int) -> dict:
    retrieved = st.session_state.retriever.retrieve(user_query, top_k=k)
    qwen = QwenClient()

    output = {
        "user_query": user_query,
        "top_k": k,
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved],
        "retrieved_chunks": retrieved,
        "structured_facts": [],
        "time": datetime.now().isoformat(timespec="seconds"),
        "discrepancy": None,
        "novelty": None,
    }

    if use_structured_analysis:
        output["structured_facts"] = extract_structured_facts(retrieved)

    if run_discrepancy_module:
        discrepancy_prompt = build_discrepancy_prompt(user_query, retrieved)
        raw_discrepancy = qwen.generate(DISCREPANCY_SYSTEM_PROMPT, discrepancy_prompt)
        parsed_discrepancy = parse_json_response(raw_discrepancy, fallback_kind="discrepancy")
        parsed_discrepancy["user_question"] = user_query
        parsed_discrepancy["retrieved_chunk_ids"] = output["retrieved_chunk_ids"]
        parsed_discrepancy["raw_model_output"] = raw_discrepancy
        output["discrepancy"] = normalize_discrepancy_payload(parsed_discrepancy, limit=3)
        if use_heuristic_fallback and not output["discrepancy"].get("discrepancies"):
            fallback_items = []
            if use_structured_analysis:
                fallback_items = structured_discrepancies(output.get("structured_facts", []), limit=5)
            if not fallback_items:
                fallback_items = heuristic_discrepancies(user_query, retrieved, limit=3)
            if fallback_items:
                output["discrepancy"]["discrepancies"] = fallback_items
                output["discrepancy"]["discrepancies_found"] = True
                output["discrepancy"]["message"] = "Discrepancies found"
                output["discrepancy"].setdefault("answer", {})["limitations"] = "LLM returned no discrepancies; structured extraction/fallback surfaced candidate issues from retrieved chunks."

    if run_novelty_module:
        novelty_prompt = build_novelty_prompt(user_query, retrieved)
        raw_novelty = qwen.generate(NOVELTY_SYSTEM_PROMPT, novelty_prompt)
        parsed_novelty = parse_json_response(raw_novelty, fallback_kind="novelty")
        parsed_novelty["user_question"] = user_query
        parsed_novelty["retrieved_chunk_ids"] = output["retrieved_chunk_ids"]
        parsed_novelty["raw_model_output"] = raw_novelty
        output["novelty"] = normalize_novelty_payload(parsed_novelty, limit=3)
        if use_heuristic_fallback and not output["novelty"].get("novel_insights"):
            fallback_items = []
            if use_structured_analysis:
                fallback_items = structured_novel_insights(output.get("structured_facts", []), retrieved, limit=5)
            if not fallback_items:
                fallback_items = heuristic_novel_insights(user_query, retrieved, limit=3)
            if fallback_items:
                output["novelty"]["novel_insights"] = fallback_items
                output["novelty"]["novel_insights_found"] = True
                output["novelty"]["message"] = "Novel insights found"
                output["novelty"].setdefault("answer", {})["limitations"] = "LLM returned no novel insights; structured extraction/fallback surfaced candidate insights from retrieved chunks."

    return output


if run:
    counts = experiment_counts or [top_k]
    if top_k not in counts:
        counts = [top_k] + counts
    counts = sorted(set(counts))

    results = []
    for k in counts:
        with st.spinner(f"Running inconsistency + novelty analysis with top {k} chunks..."):
            try:
                results.append(run_one_query(query.strip(), k))
            except Exception as exc:
                st.error(f"Run failed for top {k}: {exc}")
                results.append({
                    "user_query": query.strip(),
                    "top_k": k,
                    "retrieved_chunk_ids": [],
                    "retrieved_chunks": [],
                    "error": str(exc),
                    "discrepancy": None,
                    "novelty": None,
                })

    st.session_state.history.append({"query": query.strip(), "results": results, "time": datetime.now().isoformat(timespec="seconds")})
    st.session_state.latest_user_query = query.strip()
    st.session_state.latest_results = results
    # Use the largest top-K result as the primary export payload.
    primary = results[-1] if results else {}
    st.session_state.latest_inconsistencies = (primary.get("discrepancy") or {}).get("discrepancies", [])
    st.session_state.latest_novel_insights = (primary.get("novelty") or {}).get("novel_insights", [])
    st.session_state.latest_structured_facts = primary.get("structured_facts", [])
    st.session_state.latest_retrieved_chunks = primary.get("retrieved_chunks", [])
    st.session_state.latest_raw_result = primary

if st.session_state.history:
    latest = st.session_state.history[-1]
    st.subheader("Latest Result")

    summary_rows = []
    for result in latest["results"]:
        d = result.get("discrepancy") or {}
        n = result.get("novelty") or {}
        summary_rows.append({
            "Top K Chunks": result.get("top_k"),
            "Retrieved Chunk IDs": ", ".join(result.get("retrieved_chunk_ids", [])),
            "Discrepancies": len(d.get("discrepancies", [])) if d else "Not run",
            "Novel Insights": len(n.get("novel_insights", [])) if n else "Not run",
            "Structured Facts": len(result.get("structured_facts", [])),
            "Status": "Error" if result.get("error") else "Completed",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    for result in latest["results"]:
        st.markdown(f"## Top {result.get('top_k')} chunk result")
        st.markdown(f"**User query:** {result.get('user_query')}")
        st.markdown(f"**Chunk IDs:** {', '.join(result.get('retrieved_chunk_ids', [])) if result.get('retrieved_chunk_ids') else '—'}")

        if result.get("error"):
            st.error(result["error"])
            continue

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Gaps / Inconsistencies", "Novel Insights", "Structured Facts", "Retrieved Chunks", "Raw JSON"])

        with tab1:
            discrepancy = result.get("discrepancy")
            if not discrepancy:
                st.info("Discrepancy module was not run.")
            else:
                answer = discrepancy.get("answer", {})
                if answer.get("text"):
                    st.markdown("**Answer based on retrieved chunks**")
                    st.write(answer.get("text"))
                discrepancies = discrepancy.get("discrepancies", [])
                if not discrepancies:
                    st.success(f"{result.get('user_query')} — No discrepancies found")
                else:
                    for item in discrepancies:
                        severity = item.get("severity", "LOW").upper()
                        header = f"{item.get('id', 'INC')} — {severity}: {item.get('title', '')}"
                        if severity == "HIGH":
                            st.error(header)
                        elif severity == "MEDIUM":
                            st.warning(header)
                        else:
                            st.info(header)
                        if item.get("risk_type"):
                            st.markdown(f"**Risk type:** {item.get('risk_type')}")
                        st.write(item.get("description", ""))
                        if item.get("assumptions_made"):
                            st.markdown("**Assumptions made:**")
                            for assump in item.get("assumptions_made", []):
                                st.write(f"- {assump}")
                        if item.get("why_it_matters"):
                            st.markdown(f"**Why it matters:** {item.get('why_it_matters')}")
                        if item.get("evidence"):
                            st.markdown("**Evidence**")
                            st.dataframe(pd.DataFrame(item.get("evidence")), use_container_width=True)
                        if item.get("suggested_resolution"):
                            st.markdown(f"**Suggested resolution:** {item.get('suggested_resolution')}")

        with tab2:
            novelty = result.get("novelty")
            if not novelty:
                st.info("Novelty module was not run.")
            else:
                answer = novelty.get("answer", {})
                if answer.get("text"):
                    st.markdown("**Answer based on retrieved chunks**")
                    st.write(answer.get("text"))
                insights = novelty.get("novel_insights", [])
                if not insights:
                    st.success(f"{result.get('user_query')} — No novel insights found")
                else:
                    for item in insights:
                        strength = item.get("strength", "LOW").upper()
                        header = f"{item.get('id', 'NOV')} — {strength}: {item.get('title', '')}"
                        if strength == "HIGH":
                            st.success(header)
                        elif strength == "MEDIUM":
                            st.warning(header)
                        else:
                            st.info(header)
                        if item.get("risk_type"):
                            st.markdown(f"**Risk type:** {item.get('risk_type')}")
                        st.write(item.get("insight", ""))
                        if item.get("assumptions_made"):
                            st.markdown("**Assumptions made:**")
                            for assump in item.get("assumptions_made", []):
                                st.write(f"- {assump}")
                        if item.get("why_it_matters"):
                            st.markdown(f"**Why it matters:** {item.get('why_it_matters')}")
                        if item.get("evidence"):
                            st.markdown("**Evidence**")
                            st.dataframe(pd.DataFrame(item.get("evidence")), use_container_width=True)
                        if item.get("suggested_follow_up"):
                            st.markdown(f"**Suggested follow-up:** {item.get('suggested_follow_up')}")

        with tab3:
            facts = result.get("structured_facts", [])
            if not facts:
                st.info("No structured facts extracted. Try increasing Top K or using a query with financial metrics such as revenue, EBITDA, PAT, debt, capex, margin, volume, or guidance.")
            else:
                fact_df = pd.DataFrame(facts)
                display_cols = ["metric", "value", "period", "scope_entity", "unit", "chunk_id", "source", "sentence"]
                st.dataframe(fact_df[[c for c in display_cols if c in fact_df.columns]], use_container_width=True)
                st.caption("This table is the key Version 3 upgrade: the app extracts metric, value, period, scope/entity, unit, source and chunk ID before comparing facts.")

        with tab4:
            chunk_df = pd.DataFrame([
                {
                    "chunk_id": c["chunk_id"],
                    "score": round(c.get("score", 0), 4),
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "text": c.get("text", "")[:900] + "...",
                }
                for c in result.get("retrieved_chunks", [])
            ])
            st.dataframe(chunk_df, use_container_width=True)

        with tab5:
            compact = {k: v for k, v in result.items() if k not in {"retrieved_chunks"}}
            st.code(json.dumps(compact, indent=2), language="json")

st.divider()
with st.expander("Conversation history"):
    for item in reversed(st.session_state.history):
        st.markdown(f"**{item['time']}** — {item['query']}")


# ============================================================
# VALIDATION GUIDE + DOWNLOAD EXPORTS
# ============================================================

def _safe_df(data):
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in data):
            return pd.json_normalize(data)
        return pd.DataFrame({"value": data})
    if isinstance(data, dict):
        return pd.json_normalize(data)
    return pd.DataFrame({"value": [str(data)]})

def _excel_bytes(payload: dict):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary = pd.DataFrame([{
            "User Query": payload.get("user_query", ""),
            "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Inconsistencies": len(payload.get("inconsistencies", [])),
            "Novel Insights": len(payload.get("novel_insights", [])),
            "Structured Facts": len(payload.get("structured_facts", [])),
            "Retrieved Chunks": len(payload.get("retrieved_chunks", [])),
        }])
        sheets = {
            "Summary": summary,
            "Inconsistencies": _safe_df(payload.get("inconsistencies", [])),
            "Novel Insights": _safe_df(payload.get("novel_insights", [])),
            "Structured Facts": _safe_df(payload.get("structured_facts", [])),
            "Retrieved Chunks": _safe_df(payload.get("retrieved_chunks", [])),
            "All TopK Results": _safe_df(payload.get("all_results", [])),
        }
        for name, df in sheets.items():
            if df.empty:
                df = pd.DataFrame([{"Message": "No data available"}])
            df = df.applymap(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            df.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output

def _markdown_report(payload: dict):
    lines = [
        "# Cortexa Gap / Inconsistency and Novel Insight Report", "",
        f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        f"**User Query:** {payload.get('user_query', '')}", "",
        "## Summary", "",
        f"- Inconsistencies / gaps: {len(payload.get('inconsistencies', []))}",
        f"- Novel insights: {len(payload.get('novel_insights', []))}",
        f"- Structured facts: {len(payload.get('structured_facts', []))}",
        f"- Retrieved chunks: {len(payload.get('retrieved_chunks', []))}", "",
        "## Inconsistencies / Gaps", ""
    ]
    for item in payload.get("inconsistencies", []):
        lines += [f"### {item.get('id','INC')} — {item.get('title','')}", "",
                  f"**Severity:** {item.get('severity','')}",
                  f"**Risk type:** {item.get('risk_type','')}",
                  f"**Finding type:** {item.get('type','')}", "",
                  f"**Description:** {item.get('description','')}", ""]
        if item.get("assumptions_made"):
            lines += ["**Assumptions made:**"] + [f"- {a}" for a in item.get("assumptions_made", [])] + [""]
        if item.get("evidence"):
            lines += ["**Evidence:**"]
            for ev in item.get("evidence", []):
                lines.append(f"- {ev.get('chunk_id','')}: {ev.get('quote_or_phrase','')}")
            lines.append("")
        lines += [f"**Why it matters:** {item.get('why_it_matters','')}", f"**Suggested resolution:** {item.get('suggested_resolution','')}", ""]
    if not payload.get("inconsistencies"):
        lines += ["No inconsistencies/gaps detected.", ""]
    lines += ["## Novel Insights", ""]
    for item in payload.get("novel_insights", []):
        lines += [f"### {item.get('id','NOV')} — {item.get('title','')}", "",
                  f"**Strength:** {item.get('strength','')}",
                  f"**Risk type:** {item.get('risk_type','')}",
                  f"**Category:** {item.get('category','')}", "",
                  f"**Insight:** {item.get('insight','')}", ""]
        if item.get("assumptions_made"):
            lines += ["**Assumptions made:**"] + [f"- {a}" for a in item.get("assumptions_made", [])] + [""]
        if item.get("evidence"):
            lines += ["**Evidence:**"]
            for ev in item.get("evidence", []):
                lines.append(f"- {ev.get('chunk_id','')}: {ev.get('quote_or_phrase','')}")
            lines.append("")
        lines += [f"**Why it matters:** {item.get('why_it_matters','')}", f"**Suggested follow-up:** {item.get('suggested_follow_up','')}", ""]
    if not payload.get("novel_insights"):
        lines += ["No novel insights detected.", ""]
    return "\n".join(lines)

st.divider()
st.subheader("Download Analysis")
if "latest_raw_result" not in st.session_state:
    st.info("Run an analysis first. Download buttons will appear after the first completed run.")
else:
    export_payload = {
        "user_query": st.session_state.get("latest_user_query", ""),
        "inconsistencies": st.session_state.get("latest_inconsistencies", []),
        "novel_insights": st.session_state.get("latest_novel_insights", []),
        "structured_facts": st.session_state.get("latest_structured_facts", []),
        "retrieved_chunks": st.session_state.get("latest_retrieved_chunks", []),
        "all_results": st.session_state.get("latest_results", []),
        "raw_result": st.session_state.get("latest_raw_result", {}),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("Download Excel", data=_excel_bytes(export_payload), file_name=f"cortexa_analysis_{ts}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with col_b:
        st.download_button("Download JSON", data=json.dumps(export_payload, indent=2, ensure_ascii=False, default=str), file_name=f"cortexa_analysis_{ts}.json", mime="application/json", use_container_width=True)
    with col_c:
        st.download_button("Download Markdown", data=_markdown_report(export_payload), file_name=f"cortexa_analysis_{ts}.md", mime="text/markdown", use_container_width=True)

with st.expander("3-domain validation plan"):
    st.markdown("""
Use the same workflow across at least three domains and compare app output with Claude/ChatGPT output.

1. **Finance:** annual report + investor presentation + broker/ratings note. Test revenue, EBITDA, PAT, margins, debt, capex, guidance, and narrative vs metrics.  
2. **Clinical / Pharma:** FDA briefing + sponsor briefing + trial/public summary. Test efficacy, safety, endpoints, population, adverse events, and regulatory interpretation.  
3. **Policy / ESG / Climate:** report + action plan + resolution/implementation note. Test commitments, timelines, financing, accountability, implementation gaps, and stated risks.

For each domain, record: documents used, query, top-K, app findings, Claude/ChatGPT findings, missed issues, false positives, and next fix.
""")
