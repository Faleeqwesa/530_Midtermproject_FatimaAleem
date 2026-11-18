# CDC Mental Health Explorer (Streamlit app)

This app extends the existing midterm portfolio without replacing it. The existing `index.html` remains the GitHub Pages portfolio; this folder-level app runs separately with Streamlit.

## What it does
1. **Explore data:** filter CDC Mental Health rows by question, unit, estimate type, year, location, and demographic group; view trends, coverage, and downloadable rows.
2. **AI agent workflow:** run three inspectable data tools in order—retrieve records, check missing values/year coverage, and summarize comparable values—then generate an evidence-grounded explanation. If no API key is configured, a transparent rule-based explanation is shown.
3. **Traditional ML:** run K-Means on each location's mean annual value and year-to-year variation. It requires one comparable question/unit/estimate/demographic stratum and at least three locations with at least two years.

## Run in Codex or locally
1. Open this repository in Codex.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Upload `U.S._Chronic_Disease_Indicators_20260930.csv` in the sidebar, or place it beside `app.py`.

The CSV is approximately 127 MB and is intentionally not committed to GitHub. It can be supplied at runtime.

## Optional LLM setup
Create a local `.env` file from `.env.example` and set `OPENAI_API_KEY`. Keep the real key out of Git. You can set `OPENAI_MODEL` if your account uses a different model. For Streamlit Community Cloud, add the key in the app's Secrets settings.

Without a key, the retrieval, quality-check, and analysis workflow still runs; the final explanation uses a rule-based summary.

## Tests
```bash
pytest -q
```

## Important interpretation rules
- The app filters the unit using `DataValueUnit` (Number or %), and the estimate using `DataValueType` (for example, Crude Prevalence or Age-adjusted Prevalence). These are different columns.
- Missing values are not zero.
- K-Means cluster numbers are arbitrary group labels, not rankings.
- CDC indicators describe populations and cannot diagnose or explain an individual's mental health.
