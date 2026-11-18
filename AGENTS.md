# Codex project instructions

## Project
This repository contains a static GitHub Pages portfolio for a CDC Mental Health data investigation. The Streamlit app is a separate application in the repository root. Preserve the existing `index.html`, charts, and portfolio content unless a task explicitly asks to change them.

## Data and domain rules
- The source is the CDC U.S. Chronic Disease Indicators CSV. Filter to `Topic == "Mental Health"`.
- Important columns include `Question`, `YearStart`, `LocationDesc`, `DataValue`, `DataValueUnit`, `DataValueType`, `StratificationCategory1`, and `Stratification1`.
- `DataValueUnit` means Number or %. Do not confuse it with `DataValueType`, which distinguishes estimate types such as Crude Prevalence, Age-adjusted Prevalence, Crude Mean, and Age-adjusted Mean.
- Never treat missing CDC values as zero.
- Do not aggregate across different questions, units, estimate types, or demographic strata as if they were directly comparable.
- Do not make causal claims, individual diagnoses, or treatment recommendations.

## Architecture
- Keep data processing in `cdc_analysis.py`; keep Streamlit UI in `app.py`.
- The agent workflow is intentionally inspectable: retrieve, quality-check, analyze, then explain.
- K-Means must use a single comparable measure and retain the existing two features: mean annual value and year-to-year variation. Keep deterministic settings (`random_state=42`, `n_init=10`).
- API keys must come from environment variables or Streamlit Secrets. Never commit credentials or place keys in frontend code.

## Development
- Install dependencies with `pip install -r requirements.txt`.
- Run locally with `streamlit run app.py`.
- Run tests with `pytest -q`.
- Do not commit the 127 MB source CSV; users upload it or provide `CDC_DATA_PATH`.
- Keep explanations understandable to a student who needs to explain the code and methods in a class presentation. Prefer small, commented functions over unnecessary abstractions.
