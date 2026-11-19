"""
Maya Raghavan, Mahima Rajan, Rishali Vuriti
CSE 163 Autumn 2026
Opioid Dispensing and Overdose Mortality EDA - Test File

This file tests the merging function used in the exploratory
data analysis with small sample DataFrames.
"""

import pandas as pd

from rq1_eda import merge_data


def test_merge_data() -> None:
    """
    Tests whether mortality and dispensing data merge correctly.

    Creates small sample DataFrames containing matching and
    nonmatching state-year observations. Checks that only
    matching records are included, their values are preserved,
    and no duplicate state-year observations are created.
    """
    mortality = pd.DataFrame({
        'YEAR': [2020, 2021, 2020],
        'Location': ['Alabama', 'Alabama', 'Alaska'],
        'Death Rate': [20.0, 25.0, 30.0]
    })

    dispensing = pd.DataFrame({
        'Year': [2020, 2021, 2022],
        'Location': ['Alabama', 'Alabama', 'Alaska'],
        'Opioid Dispensing Rate': [50.0, 45.0, 40.0]
    })

    result = merge_data(mortality, dispensing)

    assert len(result) == 2
    assert result['Year'].tolist() == [2020, 2021]
    assert result['Death Rate'].tolist() == [20.0, 25.0]
    assert result['Opioid Dispensing Rate'].tolist() == [50.0, 45.0]
    assert result.duplicated(subset=['Location', 'Year']).sum() == 0


def test_merge_state_whitespace() -> None:
    """
    Tests whether state names with extra whitespace merge correctly.

    Creates sample DataFrames where one state name contains
    leading and trailing whitespace. Checks that the merge
    removes the whitespace and matches the state correctly.
    """
    mortality = pd.DataFrame({
        'YEAR': [2020],
        'Location': [' Alabama '],
        'Death Rate': [20.0]
    })

    dispensing = pd.DataFrame({
        'Year': [2020],
        'Location': ['Alabama'],
        'Opioid Dispensing Rate': [50.0]
    })

    result = merge_data(mortality, dispensing)

    assert len(result) == 1
    assert result.loc[0, 'Location'] == 'Alabama'


def main() -> None:
    """
    Runs all tests for the EDA merging function.

    Calls each testing function and prints a success message
    if all assertions pass.
    """
    test_merge_data()
    test_merge_state_whitespace()
    print('All tests passed!')


if __name__ == '__main__':
    main()
