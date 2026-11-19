"""
Rishali Vuriti, Mahima Rajan, Maya Raghavan
CSE 163 A
Research Question 2

This program tests for rq2_eda.py
"""

import pandas as pd
from rq2_eda import clean_data, combine_data


def test_clean_data() -> None:
    """
    Test that clean_data keeps and renames the correct columns.
    """
    mortality = pd.DataFrame({
        "YEAR": [2023],
        "Location": ["Washington"],
        "Death Rate": [25.0],
        "Deaths": [1000],
        "URL": ["example.com"]
    })

    opioid = pd.DataFrame({
        "Year": [2023],
        "Location": ["Washington"],
        "STATE_ABBREV": ["WA"],
        "STATE_FIPS": [53],
        "Opioid Dispensing Rate": [40.0],
        "Opioid Dispensing Rate (per 100 persons)": [40.0]
    })

    naloxone = pd.DataFrame({
        "Year": [2023],
        "Location": ["Washington"],
        "STATE_ABBREV": ["WA"],
        "STATE_FIPS": [53],
        "Naloxone Dispensing Rate": [0.5],
        "Naloxone Dispensing Rate (per 100 persons)": [0.5]
    })

    mortality, opioid, naloxone = clean_data(
        mortality, opioid, naloxone
    )

    assert "Year" in mortality.columns
    assert "YEAR" not in mortality.columns
    assert "URL" not in mortality.columns
    assert "Opioid Dispensing Rate" in opioid.columns
    assert "Naloxone Dispensing Rate" in naloxone.columns


def test_combine_data() -> None:
    """
    Test that combine_data correctly merges the datasets.
    """
    mortality = pd.DataFrame({
        "Year": [2023, 2023],
        "Location": ["Washington", "Oregon"],
        "Death Rate": [25.0, 30.0],
        "Deaths": [1000, 800]
    })

    opioid = pd.DataFrame({
        "Year": [2023, 2023],
        "Location": ["Washington", "Oregon"],
        "Opioid Dispensing Rate": [40.0, 35.0]
    })

    naloxone = pd.DataFrame({
        "Year": [2023],
        "Location": ["Washington"],
        "Naloxone Dispensing Rate": [0.5]
    })

    combined = combine_data(mortality, opioid, naloxone)

    assert len(combined) == 1
    assert combined.iloc[0]["Location"] == "Washington"
    assert combined.iloc[0]["Death Rate"] == 25.0
    assert combined.iloc[0]["Opioid Dispensing Rate"] == 40.0
    assert combined.iloc[0]["Naloxone Dispensing Rate"] == 0.5


def test_combined_data_columns() -> None:
    """
    Test that the combined dataset has the expected columns.
    """
    mortality = pd.DataFrame({
        "Year": [2023],
        "Location": ["Washington"],
        "Death Rate": [25.0],
        "Deaths": [1000]
    })

    opioid = pd.DataFrame({
        "Year": [2023],
        "Location": ["Washington"],
        "Opioid Dispensing Rate": [40.0]
    })

    naloxone = pd.DataFrame({
        "Year": [2023],
        "Location": ["Washington"],
        "Naloxone Dispensing Rate": [0.5]
    })

    combined = combine_data(mortality, opioid, naloxone)

    assert list(combined.columns) == [
        "Year",
        "Location",
        "Death Rate",
        "Deaths",
        "Opioid Dispensing Rate",
        "Naloxone Dispensing Rate"
    ]


def main() -> None:
    """
    Run all tests.
    """
    test_clean_data()
    test_combine_data()
    test_combined_data_columns()

    print("All tests passed!")


if __name__ == "__main__":
    main()
