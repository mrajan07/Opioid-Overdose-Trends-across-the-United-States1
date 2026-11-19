"""
Rishali Vuriti, Mahima Rajan, Maya Raghavan
CSE 163 A
Research Question 2

This program explores the relationship between opioid dispensing,
naloxone dispensing, and drug overdose mortality across states.
It combines the three datasets to investigate whether naloxone
dispensing rates are associated with overdose death rates.
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three original datasets.
    """
    mortality = pd.read_csv("Drug Overdose Mortality by State.csv")
    opioid = pd.read_csv("State Opioid Dispensing Rates.csv")
    naloxone = pd.read_csv("State Naloxone Dispensing Rates.csv")

    return mortality, opioid, naloxone


def clean_data(
    mortality: pd.DataFrame,
    opioid: pd.DataFrame,
    naloxone: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep relevant columns and standardize column names.
    """
    mortality = mortality.rename(columns={"YEAR": "Year"})

    mortality = mortality[
        ["Year", "Location", "Death Rate", "Deaths"]
    ]

    opioid = opioid[
        ["Year", "Location", "Opioid Dispensing Rate"]
    ]

    naloxone = naloxone[
        ["Year", "Location", "Naloxone Dispensing Rate"]
    ]

    return mortality, opioid, naloxone


def combine_data(
    mortality: pd.DataFrame,
    opioid: pd.DataFrame,
    naloxone: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all three datasets using year and location.
    """
    combined = mortality.merge(
        opioid,
        on=["Year", "Location"],
        how="inner"
    )

    combined = combined.merge(
        naloxone,
        on=["Year", "Location"],
        how="inner"
    )

    return combined


def save_data(combined: pd.DataFrame) -> None:
    """
    Save the combined dataset to a CSV file.
    """
    combined.to_csv("rq2_combined.csv", index=False)


def inspect_data(combined: pd.DataFrame) -> None:
    """
    Print information about the combined dataset.
    """
    print("\nCombined Dataset Shape:")
    print(combined.shape)

    print("\nFirst Five Rows:")
    print(combined.head())

    print("\nColumn Information:")
    combined.info()

    print("\nMissing Values:")
    print(combined.isna().sum())

    print("\nDuplicate State-Year Rows:")
    print(combined.duplicated(
        subset=["Year", "Location"]
    ).sum())

    print("\nYears:")
    print(sorted(combined["Year"].unique()))

    print("\nNumber of Locations:")
    print(combined["Location"].nunique())


def summarize_data(combined: pd.DataFrame) -> None:
    """
    Print summary statistics for variables of interest.
    """
    print("\nSummary Statistics:")

    columns = [
        "Death Rate",
        "Deaths",
        "Opioid Dispensing Rate",
        "Naloxone Dispensing Rate"
    ]

    print(combined[columns].describe())


def summarize_by_year(combined: pd.DataFrame) -> None:
    """
    Print average rates for each year.
    """
    yearly_means = combined.groupby("Year")[
        [
            "Death Rate",
            "Opioid Dispensing Rate",
            "Naloxone Dispensing Rate"
        ]
    ].mean()

    print("\nAverage Rates by Year:")
    print(yearly_means)


def scatterplot_relationship(combined: pd.DataFrame) -> None:
    """
    Plot the relationship between all three rates.
    """
    plt.figure()

    scatter = plt.scatter(
        combined["Naloxone Dispensing Rate"],
        combined["Death Rate"],
        c=combined["Opioid Dispensing Rate"]
    )

    plt.xlabel("Naloxone Dispensing Rate")
    plt.ylabel("Drug Overdose Death Rate")
    plt.title("Naloxone Dispensing and Overdose Mortality")

    plt.colorbar(
        scatter,
        label="Opioid Dispensing Rate"
    )

    plt.savefig("naloxone_mortality_scatter.png")


def lineplot_by_year(combined: pd.DataFrame) -> None:
    """
    Plot relative changes in average rates over time.
    """
    yearly_means = combined.groupby("Year")[
        [
            "Death Rate",
            "Opioid Dispensing Rate",
            "Naloxone Dispensing Rate"
        ]
    ].mean()

    indexed_means = yearly_means / yearly_means.loc[2019] * 100

    plt.figure()

    plt.plot(
        indexed_means.index,
        indexed_means["Death Rate"],
        marker="o",
        label="Death Rate"
    )

    plt.plot(
        indexed_means.index,
        indexed_means["Opioid Dispensing Rate"],
        marker="o",
        label="Opioid Dispensing Rate"
    )

    plt.plot(
        indexed_means.index,
        indexed_means["Naloxone Dispensing Rate"],
        marker="o",
        label="Naloxone Dispensing Rate"
    )

    plt.xlabel("Year")
    plt.ylabel("Rate (2019 = 100)")
    plt.title("Relative Changes in Average Rates Over Time")
    plt.legend()

    plt.savefig("yearly_rate_comparison.png")


def main() -> None:
    """
    Run the data preparation and initial inspection.
    """
    mortality, opioid, naloxone = load_data()

    mortality, opioid, naloxone = clean_data(
        mortality, opioid, naloxone
    )

    combined = combine_data(mortality, opioid, naloxone)

    save_data(combined)
    inspect_data(combined)
    summarize_data(combined)
    summarize_by_year(combined)
    scatterplot_relationship(combined)
    lineplot_by_year(combined)


if __name__ == "__main__":
    main()
