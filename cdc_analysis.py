"""Data processing for the CDC Mental Health Explorer.

The functions in this module are deliberately separate from Streamlit so they
can be tested and reused by the app.
"""
from __future__ import annotations

from typing import Any
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

REQUIRED_COLUMNS = {
    "Topic", "Question", "YearStart", "LocationDesc", "DataValue",
    "DataValueUnit", "DataValueType", "StratificationCategory1",
    "Stratification1",
}

FILTER_COLUMNS = {
    "question": "Question",
    "data_unit": "DataValueUnit",
    "estimate_type": "DataValueType",
    "locations": "LocationDesc",
    "demographic_categories": "StratificationCategory1",
    "demographic_groups": "Stratification1",
}


def prepare_mental_health_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate, copy, and prepare only the Mental Health topic."""
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError("The uploaded CSV is missing required columns: " + ", ".join(sorted(missing)))

    data = raw.loc[raw["Topic"].astype(str).str.strip().eq("Mental Health")].copy()
    data["YearStart"] = pd.to_numeric(data["YearStart"], errors="coerce")
    data["DataValue"] = pd.to_numeric(data["DataValue"], errors="coerce")
    for column in ("Question", "LocationDesc", "DataValueUnit", "DataValueType",
                   "StratificationCategory1", "Stratification1"):
        data[column] = data[column].fillna("Not reported").astype(str).str.strip()
    return data


def filter_mental_health_data(data: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply the user's selections without turning missing values into zero."""
    result = data.copy()
    for key, column in FILTER_COLUMNS.items():
        selected = filters.get(key)
        if selected is None or selected == "All":
            continue
        if isinstance(selected, (list, tuple, set)):
            selected = list(selected)
            if not selected or "All" in selected:
                continue
            result = result[result[column].isin(selected)]
        else:
            result = result[result[column].eq(str(selected))]

    start_year = filters.get("start_year")
    end_year = filters.get("end_year")
    if start_year is not None:
        result = result[result["YearStart"] >= int(start_year)]
    if end_year is not None:
        result = result[result["YearStart"] <= int(end_year)]
    return result.copy()


def retrieve_cdc_data(data: pd.DataFrame) -> dict[str, Any]:
    """Agent stage 1: retrieve matching rows and summarize availability."""
    preview_columns = [
        "LocationDesc", "YearStart", "Question", "DataValue", "DataValueUnit",
        "DataValueType", "StratificationCategory1", "Stratification1",
        "DataValueFootnote",
    ]
    preview_columns = [c for c in preview_columns if c in data.columns]
    preview = data[preview_columns].head(50).copy()
    preview = preview.astype(object).where(pd.notna(preview), None)
    return {
        "matching_rows": int(len(data)),
        "rows_with_values": int(data["DataValue"].notna().sum()),
        "rows_missing_values": int(data["DataValue"].isna().sum()),
        "records": preview.to_dict("records"),
    }


def check_cdc_data_quality(data: pd.DataFrame, filters: dict[str, Any]) -> dict[str, Any]:
    """Agent stage 2: report missing values and year coverage by location."""
    locations = filters.get("locations") or sorted(data["LocationDesc"].unique().tolist())
    years_by_location: dict[str, list[int]] = {}
    for location in locations:
        years = data.loc[data["LocationDesc"].eq(location), "YearStart"].dropna()
        years_by_location[str(location)] = sorted(years.astype(int).unique().tolist())

    year_sets = [set(v) for v in years_by_location.values()]
    shared = sorted(set.intersection(*year_sets)) if year_sets else []
    start = int(filters.get("start_year", int(data["YearStart"].min()))) if len(data) else 0
    end = int(filters.get("end_year", int(data["YearStart"].max()))) if len(data) else -1
    return {
        "matching_rows": int(len(data)),
        "rows_with_values": int(data["DataValue"].notna().sum()),
        "rows_missing_values": int(data["DataValue"].isna().sum()),
        "years_by_location": years_by_location,
        "years_shared_by_all_selected_locations": shared,
        "missing_years_by_location": {
            location: [year for year in range(start, end + 1) if year not in years]
            for location, years in years_by_location.items()
        },
        "quality_note": (
            "Missing values are not treated as zero."
            if data["DataValue"].isna().any()
            else "No missing DataValue entries were found in the selected rows."
        ),
    }


def analyze_cdc_data(data: pd.DataFrame) -> dict[str, Any]:
    """Agent stage 3: summarize numeric values by location, year, and stratum."""
    valid = data.dropna(subset=["DataValue", "YearStart"]).copy()
    group_columns = [
        "LocationDesc", "YearStart", "Question", "DataValueUnit", "DataValueType",
        "StratificationCategory1", "Stratification1",
    ]
    summary = (
        valid.groupby(group_columns, as_index=False, dropna=False, observed=True)
        .agg(mean_value=("DataValue", "mean"), number_of_records=("DataValue", "count"))
        .sort_values(["LocationDesc", "YearStart"])
    )
    return {
        "valid_rows_used": int(len(valid)),
        "summary_by_location_and_year": summary.head(500).to_dict("records"),
        "note": (
            "Values are summarized only within matching question, unit, estimate type, "
            "and demographic strata. Missing values are excluded from numeric summaries, "
            "not interpreted as zero."
        ),
    }


def run_kmeans_analysis(data: pd.DataFrame, filters: dict[str, Any]) -> dict[str, Any]:
    """Cluster locations using mean and year-to-year variation in one comparable measure."""
    for key in ("question", "data_unit", "estimate_type"):
        if not filters.get(key) or filters[key] == "All":
            raise ValueError(f"Choose one {key.replace('_', ' ')} before running K-Means.")

    categories = filters.get("demographic_categories") or []
    groups = filters.get("demographic_groups") or []
    if len(categories) != 1 or len(groups) != 1:
        raise ValueError(
            "For a fair K-Means comparison, select exactly one demographic category "
            "and one demographic group (for example, Overall / Overall)."
        )

    valid = data.dropna(subset=["YearStart", "DataValue"]).copy()
    if valid.empty:
        raise ValueError("No numeric CDC values are available for these selections.")

    annual = (
        valid.groupby(["LocationDesc", "YearStart"], as_index=False, observed=True)
        .agg(annual_value=("DataValue", "mean"))
    )
    summary = (
        annual.groupby("LocationDesc", as_index=False, observed=True)
        .agg(
            mean_value=("annual_value", "mean"),
            value_variation=("annual_value", "std"),
            years_reported=("YearStart", "nunique"),
        )
    )
    summary = summary[
        summary["years_reported"] >= 2
    ].dropna(subset=["mean_value", "value_variation"]).copy()

    if len(summary) < 3:
        raise ValueError(
            "At least three locations with two or more reported years are required. "
            "Select more locations or a wider year range."
        )

    features = ["mean_value", "value_variation"]
    scaled = StandardScaler().fit_transform(summary[features])
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    summary["cluster"] = model.fit_predict(scaled)

    score = None
    if summary["cluster"].nunique() > 1 and len(summary) > summary["cluster"].nunique():
        score = float(round(silhouette_score(scaled, summary["cluster"]), 3))

    return {
        "locations": summary.sort_values(["cluster", "mean_value"]).to_dict("records"),
        "silhouette_score": score,
        "cluster_counts": summary["cluster"].value_counts().sort_index().to_dict(),
        "features": features,
        "method_note": (
            "Each location is represented by two features: its average annual value "
            "and the standard deviation of its annual values. Features are standardized "
            "before K-Means. Cluster numbers are labels, not rankings."
        ),
    }


def build_rule_based_explanation(filters: dict[str, Any], retrieval: dict[str, Any],
                                 quality: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Fallback explanation when no LLM API key is configured."""
    rows = analysis["summary_by_location_and_year"]
    if not rows:
        pattern = "There are no numeric values available for a trend summary."
    else:
        values = [float(row["mean_value"]) for row in rows if row.get("mean_value") is not None]
        pattern = (
            f"The filtered data contains {len(values):,} grouped numeric summaries. "
            f"The smallest grouped mean is {min(values):.2f} and the largest is {max(values):.2f}. "
            "These are descriptive summaries, not evidence that one factor caused another."
            if values else "There are no numeric values available for a trend summary."
        )
    return (
        f"**Selected measure:** {filters.get('question', 'Not specified')}  \n"
        f"**Records:** {retrieval['matching_rows']:,} matched; "
        f"{retrieval['rows_with_values']:,} have numeric values and "
        f"{retrieval['rows_missing_values']:,} are missing values.  \n"
        f"**Summary:** {pattern}  \n"
        f"**Data quality:** {quality['quality_note']}  \n\n"
        "This is a population-level CDC indicator. It does not describe an individual "
        "and cannot establish causes, diagnoses, or treatment recommendations."
    )
