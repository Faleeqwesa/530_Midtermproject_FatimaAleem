"""Secure API for the CDC Mental Health agent used by the portfolio website.

The browser never receives the OpenAI API key. The API retrieves public CDC
records, runs the three inspectable data tools, and asks the LLM to explain
only the resulting evidence.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from smolagents import CodeAgent, OpenAIServerModel, tool

CDC_URL = "https://data.cdc.gov/resource/hksd-2xuw.json"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CACHE_SECONDS = 60 * 60

app = FastAPI(title="CDC Mental Health Agent API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://faleeqwesa.github.io,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_data_cache: pd.DataFrame | None = None
_data_cache_time = 0.0

FIELD_MAP = {
    "yearstart": "YearStart",
    "locationdesc": "LocationDesc",
    "topic": "Topic",
    "question": "Question",
    "datavalueunit": "DataValueUnit",
    "datavaluetype": "DataValueType",
    "datavalue": "DataValue",
    "datavaluefootnote": "DataValueFootnote",
    "stratificationcategory1": "StratificationCategory1",
    "stratification1": "Stratification1",
}


class AgentFilters(BaseModel):
    data_unit: str = "All available"
    question: str = "All questions"
    start_year: int
    end_year: int
    locations: list[str] = Field(default_factory=list)
    demographic_categories: list[str] = Field(default_factory=list)
    demographic_groups: list[str] = Field(default_factory=list)
    estimate_type: str = "All available"


def load_mental_health_data() -> pd.DataFrame:
    """Load and cache the public CDC Mental Health topic records."""
    global _data_cache, _data_cache_time
    now = time.time()
    if _data_cache is not None and now - _data_cache_time < CACHE_SECONDS:
        return _data_cache.copy()

    try:
        response = requests.get(
            CDC_URL,
            params={"$where": "topic='Mental Health'", "$limit": 50000},
            timeout=60,
        )
        response.raise_for_status()
        records = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the CDC dataset. Please try again later.",
        ) from exc

    data = pd.DataFrame(records).rename(columns=FIELD_MAP)
    required = [
        "YearStart", "LocationDesc", "Topic", "Question", "DataValue",
        "DataValueUnit", "DataValueType", "StratificationCategory1",
        "Stratification1",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise HTTPException(
            status_code=502,
            detail="The CDC response did not contain the expected dataset fields.",
        )

    data = data[data["Topic"].astype(str).str.strip().eq("Mental Health")].copy()
    data["YearStart"] = pd.to_numeric(data["YearStart"], errors="coerce")
    data["DataValue"] = pd.to_numeric(data["DataValue"], errors="coerce")
    for column in [
        "LocationDesc", "Question", "DataValueUnit", "DataValueType",
        "StratificationCategory1", "Stratification1",
    ]:
        data[column] = data[column].fillna("Not reported").astype(str).str.strip()

    _data_cache = data
    _data_cache_time = now
    return data.copy()


def filter_CDC_MH(filters: dict[str, Any]) -> pd.DataFrame:
    """Apply the selected controls to Mental Health records only."""
    data = load_mental_health_data()

    # DataValueUnit (Number / %) is deliberately separate from DataValueType
    # (for example, Crude Prevalence / Age-adjusted Prevalence).
    if filters.get("data_unit") not in (None, "", "All available"):
        data = data[data["DataValueUnit"].eq(filters["data_unit"])]

    if filters.get("question") not in (None, "", "All questions"):
        data = data[data["Question"].eq(filters["question"])]

    if filters.get("estimate_type") not in (None, "", "All available"):
        data = data[data["DataValueType"].eq(filters["estimate_type"])]

    locations = filters.get("locations") or []
    if locations:
        data = data[data["LocationDesc"].isin(locations)]

    categories = filters.get("demographic_categories") or []
    if categories and "All categories" not in categories:
        data = data[data["StratificationCategory1"].isin(categories)]

    groups = filters.get("demographic_groups") or []
    if groups and "All groups" not in groups:
        data = data[data["Stratification1"].isin(groups)]

    data = data[
        data["YearStart"].between(
            int(filters["start_year"]), int(filters["end_year"]), inclusive="both"
        )
    ]
    return data.copy()


def _json_safe_records(frame: pd.DataFrame, limit: int = 60) -> list[dict[str, Any]]:
    safe = frame.head(limit).astype(object).where(pd.notna(frame.head(limit)), None)
    return safe.to_dict("records")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "CDC Mental Health Agent API"}


@app.get("/options")
def options() -> dict[str, Any]:
    data = load_mental_health_data()
    years = sorted(data["YearStart"].dropna().astype(int).unique().tolist())
    return {
        "years": years,
        "data_units": sorted(data["DataValueUnit"].dropna().unique().tolist()),
        "questions": sorted(data["Question"].dropna().unique().tolist()),
        "locations": sorted(data["LocationDesc"].dropna().unique().tolist()),
        "demographic_categories": sorted(
            data["StratificationCategory1"].dropna().unique().tolist()
        ),
        "demographic_groups": sorted(
            data["Stratification1"].dropna().unique().tolist()
        ),
        "groups_by_category": {
            str(category): sorted(
                subset["Stratification1"].dropna().unique().tolist()
            )
            for category, subset in data.groupby("StratificationCategory1", observed=True)
        },
        "questions_by_unit": {
            str(unit): sorted(
                subset["Question"].dropna().unique().tolist()
            )
            for unit, subset in data.groupby("DataValueUnit", observed=True)
        },
        "estimate_types": sorted(data["DataValueType"].dropna().unique().tolist()),
    }


@app.post("/analyze")
def analyze(filters_model: AgentFilters) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="The AI service is not configured yet. The site owner must add the API key to the backend environment.",
        )

    filters = filters_model.model_dump()
    if filters["start_year"] > filters["end_year"]:
        raise HTTPException(status_code=400, detail="Start year must be before end year.")
    if not filters["locations"]:
        raise HTTPException(status_code=400, detail="Select at least one location.")

    selected_data = filter_CDC_MH(filters)
    if selected_data.empty:
        raise HTTPException(
            status_code=400,
            detail="No CDC records match these selections. Try widening the filters.",
        )

    # Each request gets its own result store; no user's results are shared.
    results: dict[str, dict[str, Any]] = {}

    @tool
    def retrieve_CDC_data(filters_json: str) -> str:
        """Retrieve matching CDC Mental Health rows and report value availability.

    Args:
        filters_json: JSON-encoded filter selections for the CDC records.
    """
        selected = filter_CDC_MH(json.loads(filters_json))
        columns = [
            "LocationDesc", "YearStart", "Question", "DataValue",
            "DataValueUnit", "DataValueType", "StratificationCategory1",
            "Stratification1", "DataValueFootnote",
        ]
        columns = [column for column in columns if column in selected.columns]
        result = {
            "matching_rows": int(len(selected)),
            "rows_with_values": int(selected["DataValue"].notna().sum()),
            "rows_missing_values": int(selected["DataValue"].isna().sum()),
            "records": _json_safe_records(selected[columns]),
        }
        results["retrieved"] = result
        return json.dumps(result, default=str)

    @tool
    def check_CDC_data_quality(filters_json: str) -> str:
        """Check missing values and year coverage for the selected locations.

    Args:
        filters_json: JSON-encoded filters defining locations and years.
    """
        selected_filters = json.loads(filters_json)
        selected = filter_CDC_MH(selected_filters)
        years_by_location: dict[str, list[int]] = {}
        for location in selected_filters["locations"]:
            years = selected.loc[
                selected["LocationDesc"].eq(location), "YearStart"
            ].dropna()
            years_by_location[location] = sorted(
                years.astype(int).unique().tolist()
            )
        year_sets = [set(years) for years in years_by_location.values()]
        common_years = sorted(set.intersection(*year_sets)) if year_sets else []
        result = {
            "matching_rows": int(len(selected)),
            "rows_with_values": int(selected["DataValue"].notna().sum()),
            "rows_missing_values": int(selected["DataValue"].isna().sum()),
            "years_by_location": years_by_location,
            "years_shared_by_all_selected_locations": common_years,
            "missing_years_by_location": {
                location: [
                    year
                    for year in range(
                        selected_filters["start_year"],
                        selected_filters["end_year"] + 1,
                    )
                    if year not in years_by_location[location]
                ]
                for location in selected_filters["locations"]
            },
            "quality_note": (
                "Missing values are not treated as zero."
                if selected["DataValue"].isna().any()
                else "No missing DataValue entries were found in the selected rows."
            ),
        }
        results["quality"] = result
        return json.dumps(result, default=str)

    @tool
    def analyze_CDC_data(filters_json: str) -> str:
        """Summarize numeric CDC values without combining unlike measures or groups.

    Args:
        filters_json: JSON-encoded filters defining comparable CDC records.
    """
        selected = filter_CDC_MH(json.loads(filters_json)).dropna(
            subset=["DataValue", "YearStart"]
        ).copy()
        group_columns = [
            "LocationDesc", "YearStart", "Question", "DataValueUnit",
            "DataValueType", "StratificationCategory1", "Stratification1",
        ]
        if selected.empty:
            summary = []
        else:
            summary_frame = (
                selected.groupby(
                    group_columns, as_index=False, dropna=False, observed=True
                )
                .agg(
                    mean_value=("DataValue", "mean"),
                    number_of_records=("DataValue", "count"),
                )
                .sort_values(["LocationDesc", "YearStart"])
            )
            summary = _json_safe_records(summary_frame, limit=60)
        result = {
            "valid_rows_used": int(len(selected)),
            "summary_by_location_and_year": summary,
            "note": (
                "Summaries retain question, unit, estimate type, and demographic "
                "strata separately. Missing values are excluded, not treated as zero."
            ),
        }
        results["analysis"] = result
        return json.dumps(result, default=str)

    instructions = (
        "You are an assistant analyzing CDC Mental Health data. "
        "Use the provided tools to retrieve the selected records, check data quality, "
        "and analyze the results. Call retrieve_CDC_data, check_CDC_data_quality, "
        "and analyze_CDC_data before finishing. Use the exact filters supplied. "
        "Do not invent values or years. Do not treat missing values as zero. "
        "Do not compare different questions, data-value units, estimate types, "
        "or demographic groups. If the data does not support a comparison, say so. "
        "Do not make medical diagnoses, causal claims, or treatment recommendations. "
        "Finish with final_answer."
    )
    try:
        model = OpenAIServerModel(model_id=OPENAI_MODEL, api_key=api_key)
        agent = CodeAgent(
            model=model,
            tools=[
                retrieve_CDC_data,
                check_CDC_data_quality,
                analyze_CDC_data,
            ],
            max_steps=10,
            instructions=instructions,
        )
        agent.run(
            "Use the selected CDC Mental Health filters to retrieve the data, "
            "check its quality, and analyze the results. Use all three tools "
            "before finishing. Selections: " + json.dumps(filters)
        )
        required_stages = ["retrieved", "quality", "analysis"]
        if not all(stage in results for stage in required_stages):
            raise RuntimeError("The agent did not complete all three required tools.")

        evidence = {
            "user_selections": filters,
            "retrieval_results": results["retrieved"],
            "data_quality_results": results["quality"],
            "analysis_results": results["analysis"],
        }
        explanation_instruction = (
            "Use only the supplied CDC evidence. Explain the selected question, "
            "locations, years, data unit, estimate type, and demographics. Describe "
            "patterns only when supported by the results. Mention missing values and "
            "missing years. Do not treat missing values as zero or invent statistics. "
            "Do not make causal claims, diagnoses, or treatment recommendations. "
            "Explain that these are population-level CDC indicators, not information "
            "about an individual's mental health. Keep the explanation clear and concise."
        )
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=explanation_instruction
            + "\n\nCDC results:\n"
            + json.dumps(evidence, default=str),
        )
        return {
            "filters": filters,
            "retrieval": results["retrieved"],
            "quality": results["quality"],
            "analysis": results["analysis"],
            "agent_log": "The agent completed retrieval, quality checking, and analysis.",
            "explanation": response.output_text,
            "model": OPENAI_MODEL,
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Avoid returning exception details that could reveal configuration.
        raise HTTPException(
            status_code=502,
            detail=f"The agent could not complete the request ({type(exc).__name__}). Please try again.",
        ) from exc
