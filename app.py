from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from cdc_analysis import (
    analyze_cdc_data,
    build_rule_based_explanation,
    check_cdc_data_quality,
    filter_mental_health_data,
    prepare_mental_health_data,
    retrieve_cdc_data,
    run_kmeans_analysis,
)

load_dotenv()
st.set_page_config(page_title="CDC Mental Health Explorer", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f6f8fa; }
    [data-testid="stSidebar"] { background: #eef3f5; }
    .hero { padding: 1.4rem 1.7rem; border-radius: 18px; background: #183b4e; color: white; margin-bottom: 1.2rem; }
    .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2.1rem; }
    .hero p { color: #d7e8ec; margin: 0; }
    .small-note { color: #61717b; font-size: .88rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #dbe4e8; padding: 1rem; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>CDC Mental Health Explorer</h1>
  <p>Explore U.S. mental-health indicators across years, locations, and demographic groups.</p>
</div>
""", unsafe_allow_html=True)

st.caption("A data exploration and learning tool using the CDC Chronic Disease Indicators dataset. It is not a clinical or diagnostic tool.")

DEFAULT_CSV = Path(os.getenv("CDC_DATA_PATH", "U.S._Chronic_Disease_Indicators_20260930.csv"))


@st.cache_data(show_spinner="Reading and preparing CDC data…")
def load_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    return prepare_mental_health_data(raw)


@st.cache_data(show_spinner="Reading and preparing CDC data…")
def load_from_path(path_text: str, modified_time: float) -> pd.DataFrame:
    raw = pd.read_csv(path_text, low_memory=False)
    return prepare_mental_health_data(raw)


def call_llm_explainer(filters, retrieval, quality, analysis):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return build_rule_based_explanation(filters, retrieval, quality, analysis), False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        evidence = {
            "filters": filters,
            "retrieval": {k: v for k, v in retrieval.items() if k != "records"},
            "quality": quality,
            "analysis": analysis,
        }
        response = client.responses.create(
            model=model,
            instructions=(
                "You explain CDC population-level mental-health indicator data to a general audience. "
                "Use only the supplied evidence. Do not invent values, fill missing years, or treat "
                "missing values as zero. Do not compare unlike questions, units, estimate types, or "
                "demographic strata. Describe associations or patterns without causal claims. Do not "
                "make diagnoses or treatment recommendations. State important data limitations."
            ),
            input=json.dumps(evidence, default=str),
        )
        return response.output_text, True
    except Exception as exc:
        return (
            "The LLM explanation could not be generated. The retrieval, quality-check, and "
            f"analysis steps completed. API detail: {type(exc).__name__}: {exc}\n\n"
            + build_rule_based_explanation(filters, retrieval, quality, analysis),
            False,
        )


def safe_default(options, preferred):
    return preferred if preferred in options else (options[0] if options else None)


with st.sidebar:
    st.header("1 · Load your dataset")
    uploaded = st.file_uploader("Upload the CDC CSV", type=["csv"], help="Use U.S._Chronic_Disease_Indicators_20260930.csv. The file is read in this session and is not committed to GitHub.")
    st.caption("You can also place the CSV beside app.py or set CDC_DATA_PATH.")
    st.divider()

    if uploaded is not None:
        try:
            cdc = load_from_bytes(uploaded.getvalue())
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            st.stop()
    elif DEFAULT_CSV.exists():
        try:
            cdc = load_from_path(str(DEFAULT_CSV), DEFAULT_CSV.stat().st_mtime)
        except Exception as exc:
            st.error(f"Could not read {DEFAULT_CSV.name}: {exc}")
            st.stop()
    else:
        st.info("Upload the CDC CSV to begin. The app does not include the 127 MB dataset in this repository.")
        st.stop()

    if cdc.empty:
        st.error("No rows with Topic = Mental Health were found in this CSV.")
        st.stop()

    st.success(f"Loaded {len(cdc):,} Mental Health rows")

    st.header("2 · Choose what to explore")
    questions = sorted(cdc["Question"].dropna().unique().tolist())
    default_question = safe_default(questions, "Depression among adults")
    question = st.selectbox("Question", questions, index=questions.index(default_question) if default_question else 0)

    question_data = cdc[cdc["Question"].eq(question)]
    units = sorted(question_data["DataValueUnit"].dropna().unique().tolist())
    unit_default = safe_default(units, "%")
    unit = st.selectbox("Data value unit", units, index=units.index(unit_default) if unit_default else 0)

    unit_data = question_data[question_data["DataValueUnit"].eq(unit)]
    estimate_types = sorted(unit_data["DataValueType"].dropna().unique().tolist())
    estimate_default = safe_default(estimate_types, "Crude Prevalence")
    estimate_type = st.selectbox(
        "Estimate type",
        ["All estimate types"] + estimate_types,
        index=(estimate_types.index(estimate_default) + 1) if estimate_default else 0,
        help="Keep crude and age-adjusted estimates separate when making comparisons.",
    )

    years = sorted(cdc["YearStart"].dropna().astype(int).unique().tolist())
    if not years:
        st.error("No valid years were found.")
        st.stop()
    year_range = st.slider("Year range", min_value=min(years), max_value=max(years),
                           value=(min(years), max(years)), step=1)

    locations = sorted(cdc["LocationDesc"].dropna().unique().tolist())
    default_locations = locations
    selected_locations = st.multiselect("Locations", locations, default=default_locations,
                                        help="Select one or more states, territories, or U.S. aggregate.")

    categories = sorted(cdc["StratificationCategory1"].dropna().unique().tolist())
    category_default = ["Overall"] if "Overall" in categories else categories[:1]
    selected_categories = st.multiselect("Demographic category", categories, default=category_default)

    if selected_categories:
        groups = sorted(
            cdc.loc[cdc["StratificationCategory1"].isin(selected_categories), "Stratification1"]
            .dropna().unique().tolist()
        )
    else:
        groups = sorted(cdc["Stratification1"].dropna().unique().tolist())
    group_default = ["Overall"] if "Overall" in groups else groups[:1]
    selected_groups = st.multiselect("Demographic group", groups, default=group_default)

    st.divider()
    st.caption("Tip: start with one question, one unit, one estimate type, and Overall / Overall for the clearest comparisons.")

filters = {
    "question": question,
    "data_unit": unit,
    "estimate_type": "All" if estimate_type == "All estimate types" else estimate_type,
    "start_year": int(year_range[0]),
    "end_year": int(year_range[1]),
    "locations": selected_locations,
    "demographic_categories": selected_categories,
    "demographic_groups": selected_groups,
}

filtered = filter_mental_health_data(cdc, filters)
tabs = st.tabs(["Explore data", "AI agent workflow", "K-Means patterns"])

with tabs[0]:
    st.subheader("Your filtered CDC records")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Matching records", f"{len(filtered):,}")
    metric2.metric("Numeric values", f"{filtered['DataValue'].notna().sum():,}")
    metric3.metric("Missing values", f"{filtered['DataValue'].isna().sum():,}")

    if filtered.empty:
        st.warning("No CDC records match these selections. Try widening the year range or selecting more locations/groups.")
    else:
        valid = filtered.dropna(subset=["DataValue", "YearStart"]).copy()
        group_cols = ["YearStart", "LocationDesc", "StratificationCategory1", "Stratification1", "DataValueType"]
        trend = valid.groupby(group_cols, as_index=False, observed=True).agg(
            mean_value=("DataValue", "mean"), records=("DataValue", "count")
        )
        if not trend.empty:
            color_col = "LocationDesc" if len(selected_locations) <= 8 else "Stratification1"
            st.plotly_chart(
                px.line(trend, x="YearStart", y="mean_value", color=color_col,
                        markers=True, hover_data=["LocationDesc", "StratificationCategory1",
                                                  "Stratification1", "DataValueType", "records"],
                        labels={"YearStart": "Year", "mean_value": f"Mean ({unit})",
                                "LocationDesc": "Location", "Stratification1": "Group"},
                        title="Reported mental-health indicator over time"),
                use_container_width=True,
            )
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("**Records by year**")
            by_year = valid.groupby("YearStart", as_index=False).agg(records=("DataValue", "count"))
            st.bar_chart(by_year.set_index("YearStart"))
        with right:
            st.markdown("**Data coverage**")
            coverage = valid.groupby("LocationDesc", as_index=False).agg(
                years_reported=("YearStart", "nunique"), records=("DataValue", "count")
            ).sort_values("records", ascending=False)
            st.dataframe(coverage, use_container_width=True, hide_index=True)

        st.markdown("**Filtered records (first 100 rows)**")
        display_cols = [c for c in [
            "LocationDesc", "YearStart", "Question", "DataValue", "DataValueUnit",
            "DataValueType", "StratificationCategory1", "Stratification1", "DataValueFootnote"
        ] if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(100), use_container_width=True, hide_index=True)
        csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered records", csv_bytes, "cdc_mental_health_filtered.csv", "text/csv")

    st.caption("A missing CDC value is not zero. This view describes reported population-level indicators; it does not establish why a pattern occurred.")

with tabs[1]:
    st.subheader("A three-stage agent workflow")
    st.write("The agent follows a fixed, inspectable sequence: retrieve the selected records → check data quality → analyze comparable values → explain the evidence.")
    if st.button("Run agent analysis", type="primary", key="run_agent"):
        if not selected_locations:
            st.error("Select at least one location before running the agent.")
        elif not selected_categories or not selected_groups:
            st.error("Select at least one demographic category and group.")
        elif filtered.empty:
            st.error("No records match these filters.")
        else:
            with st.spinner("Running retrieval, quality checks, and analysis…"):
                retrieval = retrieve_cdc_data(filtered)
                quality = check_cdc_data_quality(filtered, filters)
                analysis = analyze_cdc_data(filtered)
                explanation, used_llm = call_llm_explainer(filters, retrieval, quality, analysis)
            st.success("All three data tools completed.")
            st.markdown("**Step 1 — Retrieve CDC records**")
            st.write(f"{retrieval['matching_rows']:,} matching rows; {retrieval['rows_with_values']:,} with values; {retrieval['rows_missing_values']:,} missing.")
            st.dataframe(pd.DataFrame(retrieval["records"]).head(15), use_container_width=True, hide_index=True)
            st.markdown("**Step 2 — Check data quality**")
            q1, q2 = st.columns(2)
            q1.metric("Locations checked", len(quality["years_by_location"]))
            q2.metric("Years shared by all selected locations", len(quality["years_shared_by_all_selected_locations"]))
            st.write(quality["quality_note"])
            with st.expander("View year coverage by location"):
                st.json(quality["years_by_location"])
                st.json(quality["missing_years_by_location"])
            st.markdown("**Step 3 — Analyze the selected records**")
            st.write(f"{analysis['valid_rows_used']:,} numeric rows used.")
            st.dataframe(pd.DataFrame(analysis["summary_by_location_and_year"]).head(100),
                         use_container_width=True, hide_index=True)
            st.markdown("**Final explanation**")
            st.caption("LLM-generated explanation" if used_llm else "Rule-based explanation (add OPENAI_API_KEY to enable the LLM explanation)")
            st.markdown(explanation)
            st.download_button("Download agent evidence (JSON)",
                               json.dumps({"filters": filters, "retrieval": retrieval,
                                           "quality": quality, "analysis": analysis},
                                          default=str, indent=2),
                               "cdc_agent_evidence.json", "application/json")

with tabs[2]:
    st.subheader("K-Means: group locations with similar patterns")
    st.write("This follows your notebook's two-feature approach: average annual value and variation across years. K-Means groups locations; it does not rank them.")
    st.info("For a valid comparison, select one question, one data unit, one estimate type, and exactly one demographic category/group. Select at least three locations with two or more years of data.")
    if st.button("Run K-Means clustering", type="primary", key="run_kmeans"):
        try:
            if estimate_type == "All estimate types":
                raise ValueError("Choose one estimate type in the sidebar before running K-Means.")
            if not selected_locations:
                raise ValueError("Select at least three locations in the sidebar.")
            result = run_kmeans_analysis(filtered, filters)
            cluster_df = pd.DataFrame(result["locations"])
            st.metric("Silhouette score", "Not available" if result["silhouette_score"] is None else result["silhouette_score"])
            st.caption(result["method_note"])
            st.plotly_chart(
                px.scatter(cluster_df, x="mean_value", y="value_variation", color="cluster",
                           hover_name="LocationDesc", hover_data=["years_reported"],
                           labels={"mean_value": f"Mean annual value ({unit})",
                                   "value_variation": "Year-to-year variation", "cluster": "Cluster"},
                           title="K-Means location clusters"),
                use_container_width=True,
            )
            st.markdown("**Cluster assignments**")
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            st.markdown("**Locations in each cluster**")
            st.json({str(k): int(v) for k, v in result["cluster_counts"].items()})
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"K-Means could not complete: {type(exc).__name__}: {exc}")

st.divider()
st.caption("Source: CDC U.S. Chronic Disease Indicators. Data availability varies by year, location, measure, and demographic group. Review CDC footnotes and definitions before drawing conclusions.")
