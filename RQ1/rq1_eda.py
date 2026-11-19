"""
Maya Raghavan, Mahima Rajan, Rishali Vuriti
CSE 163 Autumn 2026
Opioid Dispensing and Overdose Mortality EDA (Research Question 1)

This file analyzes the relationship between opioid dispensing rates
and drug overdose mortality rates across U.S. states from 2019 to 2024.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


MORTALITY_FILE = 'Drug Overdose Mortality by State.csv'
DISPENSING_FILE = 'State Opioid Dispensing Rates.csv'
OUTPUT_FOLDER = 'eda_results'
MERGED_FILE = 'rq1_merged_data_file.csv'


def load_data(mortality_file: str,
              dispensing_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the mortality and opioid dispensing CSV files.

    Takes two file paths as strings and returns two pandas DataFrames,
    one containing mortality data and one containing dispensing data.
    """
    mortality = pd.read_csv(mortality_file)
    dispensing = pd.read_csv(dispensing_file)
    return mortality, dispensing


def merge_data(mortality: pd.DataFrame,
               dispensing: pd.DataFrame) -> pd.DataFrame:
    """
    Combines mortality and opioid dispensing data by state and year.

    Takes two DataFrames, removes extra whitespace from state names,
    and standardizes the year column names. Returns a new DataFrame
    containing only matching state-year observations, sorted by
    year and location.
    """
    mortality = mortality.copy()
    dispensing = dispensing.copy()

    mortality['Location'] = mortality['Location'].str.strip()
    dispensing['Location'] = dispensing['Location'].str.strip()
    mortality = mortality.rename(columns={'YEAR': 'Year'})

    merged = dispensing.merge(
        mortality,
        on=['Location', 'Year'],
        how='inner',
        validate='one_to_one'
    )

    return merged.sort_values(
        ['Year', 'Location']
    ).reset_index(drop=True)


def print_data_summary(data: pd.DataFrame, name: str) -> None:
    """
    Prints the size and missing-value information for a dataset.

    Takes a DataFrame and its name. Prints the number of rows and
    columns, the missing-value count for each column, and the
    number of rows containing at least one missing value.
    """
    print('\n' + name)
    print('Rows:', len(data))
    print('Columns:', len(data.columns))
    print('Missing values by column:')
    print(data.isna().sum())
    print('Rows with any missing values:',
          data.isna().any(axis=1).sum())


def print_statistics(merged: pd.DataFrame) -> None:
    """
    Prints descriptive statistics for the merged dataset.

    Takes a DataFrame containing opioid dispensing and mortality
    rates. Prints seven-number summaries for both quantitative
    variables, frequency counts for categorical variables, and
    the Pearson correlation between dispensing and mortality rates.
    """
    columns = ['Opioid Dispensing Rate', 'Death Rate']

    print('\nSeven-number summaries:')
    print(merged[columns].describe().loc[
        ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    ])

    print('\nYear counts:')
    print(merged['Year'].value_counts().sort_index())

    print('\nLocation counts:')
    print(merged['Location'].value_counts().sort_index())

    print('\nDispensing category counts:')
    print(merged[
        'Opioid Dispensing Rate (per 100 persons)'
    ].value_counts().sort_index())

    correlation = merged['Opioid Dispensing Rate'].corr(
        merged['Death Rate']
    )
    print('\nPearson correlation:', round(correlation, 3))


def plot_relationship(merged: pd.DataFrame) -> None:
    """
    Creates and saves a scatter plot of dispensing and mortality rates.

    Takes the merged DataFrame and plots opioid dispensing rates
    on the x-axis and drug overdose mortality rates on the y-axis.
    Saves the figure as dispensing_vs_mortality.png in the
    output folder.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        merged['Opioid Dispensing Rate'],
        merged['Death Rate'],
        alpha=0.6
    )

    ax.set_title(
        'Opioid Dispensing vs. Drug Overdose Mortality (2019–2024)'
    )
    ax.set_xlabel('Opioid prescriptions dispensed per 100 people')
    ax.set_ylabel('Age-adjusted overdose deaths per 100,000 people')

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_FOLDER, 'dispensing_vs_mortality.png')
    )
    plt.close(fig)


def plot_yearly_trends(merged: pd.DataFrame) -> None:
    """
    Creates and saves line plots of annual dispensing and mortality rates.

    Takes the merged DataFrame and calculates the average opioid
    dispensing and overdose mortality rates for each year.
    Creates two vertically arranged line plots and saves the
    figure as yearly_trends.png in the output folder.
    """
    yearly = merged.groupby('Year')[
        ['Opioid Dispensing Rate', 'Death Rate']
    ].mean()

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(
        yearly.index,
        yearly['Opioid Dispensing Rate'],
        marker='o'
    )
    axes[0].set_title('Average Opioid Dispensing Rate by Year')
    axes[0].set_ylabel('Prescriptions per 100 people')

    axes[1].plot(
        yearly.index,
        yearly['Death Rate'],
        marker='o'
    )
    axes[1].set_title('Average Drug Overdose Mortality Rate by Year')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Deaths per 100,000 people')
    axes[1].set_xticks(yearly.index)

    fig.suptitle(
        'State-Level Dispensing and Mortality Trends (2019–2024)'
    )
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_FOLDER, 'yearly_trends.png')
    )
    plt.close(fig)


def main() -> None:
    """
    Runs the exploratory data analysis on the two CDC datasets.

    Loads the datasets, prints information about their structure,
    merges matching state-year observations, calculates descriptive
    statistics, and saves the merged data and visualizations.
    """
    mortality, dispensing = load_data(
        MORTALITY_FILE, DISPENSING_FILE
    )

    print_data_summary(mortality, 'Mortality dataset')
    print_data_summary(dispensing, 'Dispensing dataset')

    merged = merge_data(mortality, dispensing)

    print_data_summary(merged, 'Merged dataset')
    print(
        'Duplicate state-year rows:',
        merged.duplicated(subset=['Location', 'Year']).sum()
    )

    print_statistics(merged)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    merged.to_csv(
        os.path.join(OUTPUT_FOLDER, MERGED_FILE),
        index=False
    )

    plot_relationship(merged)
    plot_yearly_trends(merged)

    print('\nFiles saved in:', OUTPUT_FOLDER)


if __name__ == '__main__':
    main()
