import pandas as pd
import pytest

from cdc_analysis import (
    analyze_cdc_data,
    check_cdc_data_quality,
    filter_mental_health_data,
    prepare_mental_health_data,
    retrieve_cdc_data,
    run_kmeans_analysis,
)


@pytest.fixture
def sample_data():
    rows = []
    for location, base in [("North Carolina", 10), ("Virginia", 14), ("Georgia", 18), ("Florida", 22)]:
        for year_offset, value in enumerate([base, base + 1, base + 3]):
            rows.append({
                "Topic": "Mental Health",
                "Question": "Depression among adults",
                "YearStart": 2020 + year_offset,
                "LocationDesc": location,
                "DataValue": value,
                "DataValueUnit": "%",
                "DataValueType": "Crude Prevalence",
                "StratificationCategory1": "Overall",
                "Stratification1": "Overall",
            })
    rows.append({
        "Topic": "Other topic", "Question": "Not mental health", "YearStart": 2022,
        "LocationDesc": "North Carolina", "DataValue": 99, "DataValueUnit": "%",
        "DataValueType": "Crude Prevalence", "StratificationCategory1": "Overall",
        "Stratification1": "Overall",
    })
    rows.append({
        "Topic": "Mental Health", "Question": "Depression among adults", "YearStart": 2022,
        "LocationDesc": "North Carolina", "DataValue": None, "DataValueUnit": "%",
        "DataValueType": "Crude Prevalence", "StratificationCategory1": "Overall",
        "Stratification1": "Overall",
    })
    return pd.DataFrame(rows)


def test_prepare_filters_to_mental_health_and_keeps_missing(sample_data):
    prepared = prepare_mental_health_data(sample_data)
    assert len(prepared) == 13
    assert prepared["DataValue"].isna().sum() == 1


def test_unit_filter_uses_data_value_unit(sample_data):
    prepared = prepare_mental_health_data(sample_data)
    selected = filter_mental_health_data(prepared, {
        "question": "Depression among adults",
        "data_unit": "%",
        "estimate_type": "Crude Prevalence",
        "locations": ["North Carolina"],
        "demographic_categories": ["Overall"],
        "demographic_groups": ["Overall"],
        "start_year": 2020,
        "end_year": 2022,
    })
    assert len(selected) == 4


def test_agent_stages_report_missing_data(sample_data):
    prepared = prepare_mental_health_data(sample_data)
    selected = filter_mental_health_data(prepared, {"question": "Depression among adults"})
    retrieved = retrieve_cdc_data(selected)
    quality = check_cdc_data_quality(selected, {"locations": ["North Carolina"], "start_year": 2020, "end_year": 2022})
    analysis = analyze_cdc_data(selected)
    assert retrieved["rows_missing_values"] == 1
    assert quality["rows_missing_values"] == 1
    assert analysis["valid_rows_used"] == 12


def test_kmeans_requires_one_comparable_stratum(sample_data):
    prepared = prepare_mental_health_data(sample_data)
    selected = filter_mental_health_data(prepared, {"question": "Depression among adults"})
    with pytest.raises(ValueError, match="exactly one demographic"):
        run_kmeans_analysis(selected, {
            "question": "Depression among adults", "data_unit": "%", "estimate_type": "Crude Prevalence",
            "demographic_categories": ["Overall", "Sex"], "demographic_groups": ["Overall"],
        })


def test_kmeans_runs_for_comparable_data(sample_data):
    prepared = prepare_mental_health_data(sample_data)
    selected = filter_mental_health_data(prepared, {"question": "Depression among adults"})
    result = run_kmeans_analysis(selected, {
        "question": "Depression among adults", "data_unit": "%", "estimate_type": "Crude Prevalence",
        "demographic_categories": ["Overall"], "demographic_groups": ["Overall"],
    })
    assert len(result["locations"]) == 4
    assert result["silhouette_score"] is not None
