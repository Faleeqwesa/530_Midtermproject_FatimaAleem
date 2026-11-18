# CDC Mental Health Agent API

This service powers the AI Agent Explorer embedded in the existing GitHub Pages portfolio. The interface stays in `index.html`; this API keeps the OpenAI credential on the server.

## What it does

1. Retrieves the public CDC U.S. Chronic Disease Indicators records for the **Mental Health** topic.
2. Applies the website's selected data unit, question, year range, locations, demographic categories/groups, and estimate type.
3. Runs the three inspectable agent tools: `retrieve_CDC_data`, `check_CDC_data_quality`, and `analyze_CDC_data`.
4. Sends the resulting evidence to the configured OpenAI model for a constrained explanation.

Missing values are not treated as zero. The analysis keeps question, unit, estimate type, and demographic strata in the grouping keys. The service does not make medical diagnoses or causal claims.

## Deploy on Render

1. Push this branch to GitHub and open the pull request.
2. In Render, create a new Blueprint and select this repository. Render reads the root `render.yaml` file and creates the API service.
3. In the Render service's Environment settings, add `OPENAI_API_KEY` as a secret. Do not put the key in `index.html`, Git, a notebook, or a public issue.
4. Deploy. Open the service's `/health` endpoint and confirm it returns a JSON status of `ok`.
5. Copy the service's base URL, for example `https://your-service-name.onrender.com`.
6. In `index.html`, replace `https://YOUR-RENDER-SERVICE.onrender.com` in the `API_BASE` constant with the real service URL. Commit and push that change.
7. If the portfolio uses a different domain, update `FRONTEND_ORIGINS` in Render to include that exact website origin.

The API key is configured only in the backend environment. The frontend never receives it.

## Run locally

From the repository root:

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables in the terminal (do not commit a key file):

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4.1-mini"
$env:FRONTEND_ORIGINS="http://localhost:8000"
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/health` to check the API. Serve the repository's static site locally from the repository root in another terminal, then set `API_BASE` in `index.html` to `http://127.0.0.1:8000`.

## Important model note

The notebook draft used `gpt-6-astra` as a model ID. That value is not assumed to be a valid OpenAI API model identifier here. The backend defaults to `gpt-4.1-mini`; set `OPENAI_MODEL` to a model ID available to the API project if you want to use a different one.

## Data source and limits

The API requests the public Socrata CDC dataset using the Mental Health topic filter. It caches the retrieved records in memory for one hour. Render's free service may sleep while idle, so the first request after inactivity can take longer. This is an educational portfolio prototype; before wider public use, add rate limiting, request-size controls, monitoring, and a persistent cache.
