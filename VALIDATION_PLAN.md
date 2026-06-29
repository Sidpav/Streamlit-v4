# Validation Plan — Discrepancies and Novel Insights

## Boss action points covered
- Discrepancy prompt remains generic but now includes `risk_type` and `assumptions_made`.
- Implementation includes structured fact extraction and fallback checks.
- Outputs separate confirmed/potential conflicts, reconciliation issues, and novelty insights.
- Run manual benchmark with Claude/ChatGPT using the same chunks.
- Validate across three domains: Finance, Clinical/Pharma, Policy/ESG.

## Manual benchmark prompt
Paste the same retrieved chunks into Claude/ChatGPT and ask it to return: issue title, risk type, finding type, severity, evidence, assumptions made, why it matters, and suggested resolution. Compare with app output.

## Validation table columns
Domain | Documents | Query | Top-K | App findings | Claude/ChatGPT findings | Missed by app | False positives | Fix required
