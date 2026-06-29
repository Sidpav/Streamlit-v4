# Cortexa Chunk Intelligence Demo — v5 risk + assumptions validation build

This build adds Risk Type, Assumptions Made, stronger structured/fallback detection, export buttons, and a 3-domain validation plan.

# Cortexa Chunk Intelligence Demo — v3 fallback-safe

# Cortexa Chunk Intelligence Demo

This Streamlit demo tests whether retrieved chunks are sufficient to surface:

1. Gaps / inconsistencies / discrepancies
2. Novel insights

The app follows a RAG-style flow:

1. User uploads one or more documents.
2. Documents are parsed, chunked, and indexed.
3. User asks a query.
4. The app retrieves the top-k chunks.
5. The same retrieved chunks are sent to two reasoning modules in parallel-style flow:
   - Discrepancy detection
   - Novelty analysis
6. The UI shows two outputs in one window:
   - User prompt + nicely articulated discrepancy with evidence
   - User prompt + nicely articulated novel insight with evidence

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Quick UI test

```bash
export LLM_BACKEND=mock
streamlit run app.py
```

Windows PowerShell:

```powershell
$env:LLM_BACKEND="mock"
streamlit run app.py
```

## Local Qwen via Ollama

```bash
ollama pull qwen2.5:3b-instruct
export LLM_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5:3b-instruct
streamlit run app.py
```

Windows PowerShell:

```powershell
$env:LLM_BACKEND="ollama"
$env:OLLAMA_MODEL="qwen2.5:3b-instruct"
streamlit run app.py
```


## GitHub + Streamlit deployment for Sidpav

Suggested GitHub repo:

```text
https://github.com/Sidpav/cortexa-chunk-intelligence-demo
```

Streamlit Cloud settings:

```text
Repository: Sidpav/cortexa-chunk-intelligence-demo
Branch: main
Main file path: app.py
```

See `DEPLOY_TO_GITHUB_AND_STREAMLIT.md` for exact push and deployment commands.

## Online deployment

For Streamlit Cloud, start with mock mode in Secrets:

```toml
LLM_BACKEND = "mock"
```

For real model output online, use API mode, for example OpenRouter:

```toml
LLM_BACKEND = "openrouter"
OPENROUTER_API_KEY = "your_key"
OPENROUTER_MODEL = "qwen/qwen-2.5-7b-instruct"
```

or Groq:

```toml
LLM_BACKEND = "groq"
GROQ_API_KEY = "your_key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

## Testing chunk sufficiency

Use the sidebar to test different retrieval counts: 10, 20, 30, 40, 50, 60.

The goal is to determine whether top-10 chunks are enough or whether meaningful discrepancies and novel insights emerge only after retrieving a larger number of chunks.

## Suggested user queries

- Are there any inconsistencies or novel insights in revenue, EBITDA, PAT, or margin trends?
- Are there discrepancies or hidden insights in management guidance versus actual performance?
- Are there inconsistencies or novel insights around cash generation, debt, capex, or profitability?
- Are there any contradictions or under-discussed implications in the company’s risks and outlook?

## If no inconsistencies or insights appear

This version includes a demo fallback option in the sidebar: **Use demo fallback if model returns empty output**. Keep it enabled for demos and small models. It surfaces evidence-grounded candidate discrepancies/novel insights when the LLM returns empty lists.

Why empty outputs happen:
- mock mode does not perform real reasoning;
- the retrieved top-K chunks may not contain both sides of a discrepancy;
- the user query may be too broad or too narrow;
- small models may be conservative and return no findings.

For stronger results, try top-K 20/30/40 and ask focused questions around metrics, dates, targets, guidance, valuation, risks, assumptions, or claims.


## v3 note
This version defaults to Top K 30 and experiment counts 10, 20, 30, 40. It also shows a visible `Version: v3 fallback-safe demo` caption so you can confirm Streamlit Cloud is running the latest code.

If the app still shows zero findings on the Alpha Retail test pack, the deployed Streamlit app is likely running an older GitHub commit or the fallback checkbox is disabled.

## v4 structured fact extraction upgrade
This version adds a Version 3-style analyst layer:

1. Extracts structured facts from retrieved chunks:
   - metric
   - value
   - period
   - scope/entity
   - unit
   - source
   - chunk ID
   - evidence sentence
2. Compares facts using metric + period + scope/entity + unit.
3. Separates stronger value conflicts from reconciliation/scope issues.
4. Adds a Structured Facts tab so you can see why the app flagged a finding.
5. Generates stronger novelty insights around segment mix, growth quality, cash conversion, margin quality, and balance sheet pressure.

For financial documents, keep **Use structured financial fact extraction** ON.
