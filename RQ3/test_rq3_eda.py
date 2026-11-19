"""
Project Part 2: EDA TEST
Tests for rq3_eda.py.
Mahima Rajan, Rishali Vuriti, Maya Raghavan

"""

import os
import tempfile
import pandas as pd
import rq3_eda
from rq3_eda import DEATH, NALOXONE, PREV


def test_clean_and_merge_pipeline() -> None:
    """Verify  CSVs merge, clean columns,
    compute lag rates, and drop 2017–2018."""
    naloxone = pd.DataFrame({
        "Year": [2019, 2020],
        "Location": ["Texas", " Texas"],
        "STATE_ABBREV": ["TX", "TX"],
        "STATE_FIPS": [48, 48],
        "Naloxone Dispensing Rate": [0.4, 0.6],
        "Naloxone Dispensing Rate (per 100 persons)": ["<", "<"],
    })
    mortality = pd.DataFrame({
        "YEAR": [2017, 2018, 2019, 2020],
        "Location": ["Texas", "Texas", "Texas", "Texas"],
        "Death Rate": [10.0, 12.0, 15.0, 18.0],
        "Deaths": [100, 100, 100, 100],
        "URL": ["/x.html", "/x.html", "/x.html", "/x.html"],
    })

    """
    Write temporary CSV files on disk so we also test UTF-8 BOM loading
    """
    with tempfile.TemporaryDirectory() as folder:
        nal_path = os.path.join(folder, "nal.csv")
        mort_path = os.path.join(folder, "mort.csv")
        naloxone.to_csv(nal_path, index=False, encoding="utf-8-sig")
        mortality.to_csv(mort_path, index=False, encoding="utf-8-sig")

        result = rq3_eda.build_merged_data(nal_path, mort_path)

    assert len(result) == 2
    assert list(result["state"]) == ["Texas", "Texas"]
    assert list(result["year"]) == [2019, 2020]

    row_2019 = result.iloc[0]
    assert row_2019[DEATH] == 15.0
    assert row_2019[PREV] == 12.0
    assert row_2019[NALOXONE] == 0.4

    row_2020 = result.iloc[1]
    assert row_2020[DEATH] == 18.0
    assert row_2020[PREV] == 15.0
    assert row_2020[NALOXONE] == 0.6


def test_missing_summary() -> None:
    """
    Verify missing_summary counts both NaN values and blank spaces.
    """
    df = pd.DataFrame({
        "numbers": [10.0, None],
        "text": ["   ", "valid"],
    })
    summary = rq3_eda.missing_summary(df)

    assert summary.loc["numbers", "missing_count"] == 1
    assert summary.loc["text", "missing_count"] == 1


def test_seven_number_and_categorical_summaries() -> None:
    """
    Verify summary statistics and category counts match hand calculations.
    """
    numbers_df = pd.DataFrame({"rate": [1, 2, 3, 4, 5]})
    stats = rq3_eda.seven_number_summary(numbers_df, ["rate"]).loc["rate"]

    assert stats["min"] == 1
    assert stats["Q1"] == 2
    assert stats["median"] == 3
    assert stats["Q3"] == 4
    assert stats["max"] == 5

    states_df = pd.DataFrame({"state": ["Ohio", "Alaska", "Ohio"]})
    counts = rq3_eda.categorical_summary(states_df, "state")

    assert list(counts["state"]) == ["Alaska", "Ohio"]
    assert list(counts["count"]) == [1, 2]


def test_yearly_averages() -> None:
    """
    Verify yearly_averages computes the mean across states for each year.
    """
    df = pd.DataFrame({
        "state": ["Alaska", "Hawaii"],
        "year": [2020, 2020],
        NALOXONE: [1.0, 3.0],
        DEATH: [20.0, 40.0],
    })
    avg = rq3_eda.yearly_averages(df)

    assert len(avg) == 1
    assert avg.loc[0, "year"] == 2020
    assert avg.loc[0, NALOXONE] == 2.0
    assert avg.loc[0, DEATH] == 30.0


def test_real_csv_files_if_present() -> None:
    """
    Cross-check pipeline against the actual project CSVs when available.
    """
    has_naloxone = os.path.exists(rq3_eda.NALOXONE_DISPENSING_FILE)
    has_mortality = os.path.exists(rq3_eda.MORTALITY_RATES)
    if not (has_naloxone and has_mortality):
        return

    merged = rq3_eda.build_merged_data()
    rq3_eda.check_merged(merged)

    wv_2021 = merged[
        (merged["state"] == "West Virginia") & (merged["year"] == 2021)
    ].iloc[0]
    assert wv_2021[DEATH] == 90.9
    assert wv_2021[PREV] == 81.4
    assert wv_2021[NALOXONE] == 0.7


if __name__ == "__main__":
    all_tests = [
        test_clean_and_merge_pipeline,
        test_missing_summary,
        test_seven_number_and_categorical_summaries,
        test_yearly_averages,
        test_real_csv_files_if_present,
    ]
    for test_fn in all_tests:
        test_fn()
        print(f"PASS: {test_fn.__name__}")
