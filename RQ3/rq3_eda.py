"""
Project Part 2: EDA
Opioid overdose trends (Research Question 3 data).
Mahima Rajan, Rishali Vuriti, Maya Raghavan
File does everything in one run:
1. Loads the naloxone dispensing and drug overdose mortality files,
       merges them, then cleans the merged table and adds a
       previous-year mortality column.
2. Saves the final table to merged_data_file.csv.
3. Reports missing data for the raw, merged and final tables.
4. Prints summaries of the variables of interest in the merged
       data (seven-number summaries for numbers, value counts for
       categories).
5. Saves two figures:
       rq3_fig1.png  average naloxone dispensing rate and average death
                 rate per year (averaged over all 51 states)
       rq3_fig2.png  previous-year vs. current death rate, all states
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

NALOXONE_DISPENSING_FILE = "State Naloxone Dispensing Rates.csv"
MORTALITY_RATES = "Drug Overdose Mortality by State.csv"
OUTPUT_FILE = "rq3_merged_data_file.csv"

YEAR_TO_REMOVE = 2017
N_STATES = 51

DROP_COLUMNS = [
    "STATE_FIPS",
    "STATE_ABBREV",
    "URL",
    "Deaths",
    "Naloxone Dispensing Rate (per 100 persons)",
]

NALOXONE = "naloxone_dispensing_rate_per_100"
DEATH = "death_rate_per_100000"
PREV = "prev_year_death_rate_per_100000"
NALOXONE_LABEL = "Naloxone dispensing rate (per 100 persons)"
DEATH_LABEL = "Drug overdose death rate (per 100,000 persons)"
PREV_LABEL = "Previous-year death rate (per 100,000 persons)"

"""
This section of the code loads the data, cleans, and merges it.
"""


def load_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV, stripping the byte-order mark and header spaces.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the merge keys to 'year' and 'state' and clean them.
    """
    df = df.rename(columns={"Year": "year", "YEAR": "year",
                            "Location": "state"})
    df["year"] = df["year"].astype(int)
    df["state"] = df["state"].str.strip()
    return df


def merge_datasets(
    naloxone: pd.DataFrame, mortality: pd.DataFrame
) -> pd.DataFrame:
    """
    Outer-merge on state and year (keeps 2018 mortality rows).
    """
    merged = mortality.merge(naloxone, on=["state", "year"], how="outer")
    return merged


def clean_merged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unneeded columns, rename rates, filter and sort.
    """
    df = df.drop(columns=DROP_COLUMNS)
    df = df.rename(
        columns={"Naloxone Dispensing Rate": NALOXONE, "Death Rate": DEATH}
    )
    df = df[
        ["state", "year"]
        + [c for c in df.columns if c not in ("state", "year")]
    ]
    df = df[df["year"] > YEAR_TO_REMOVE]
    df = df.sort_values(["state", "year"]).reset_index(drop=True)
    return df


def add_previous_year_mortality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add each state's death rate from the year before.
    """
    prev = df[["state", "year", DEATH]].copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={DEATH: PREV})
    return df.merge(prev, on=["state", "year"], how="left")


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values (the 2018 helper rows).
    """
    return df.dropna().reset_index(drop=True)


def check_merged(df: pd.DataFrame) -> None:
    """
    Sanity checks on the final table.
    """
    assert df.shape[0] == 51 * 6
    assert df["year"].min() == 2019
    assert df["year"].max() == 2024
    assert not df.duplicated(["state", "year"]).any()
    assert not df.isna().any().any()
    assert (df[NALOXONE] >= 0).all()


def build_merged_data(
    naloxone_path: str = NALOXONE_DISPENSING_FILE,
    mortality_path: str = MORTALITY_RATES,
) -> pd.DataFrame:
    """
    Load, merge, then clean everything; return the final table.
    """
    naloxone = standardize_keys(load_csv(naloxone_path))
    mortality = standardize_keys(load_csv(mortality_path))

    merged = merge_datasets(naloxone, mortality)
    cleaned = clean_merged(merged)
    cleaned = add_previous_year_mortality(cleaned)
    return drop_incomplete_rows(cleaned)


"""
This section of the code will now check for missing data.
"""


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the count and percent of missing values per column.
    Blank or whitespace-only strings count as missing too.
    """
    blank = df.apply(
        lambda col: col.astype(str).str.strip().eq("")
        if col.dtype == object or str(col.dtype) == "str"
        else pd.Series(False, index=col.index)
    )
    missing = df.isna() | blank
    return pd.DataFrame({
        "missing_count": missing.sum(),
        "missing_percent": (missing.mean() * 100).round(2),
    })


def coverage_report(df: pd.DataFrame, name: str) -> None:
    """
    Prints which years are present and whether every state has a row
    in every present year (a gap here is missing data that isna() can't
    see, because the whole row is absent).
    """
    years: list[int] = sorted(int(y) for y in df["year"].unique())
    per_year: pd.Series = df.groupby("year")["state"].nunique()
    incomplete: pd.Series = per_year[per_year != N_STATES]
    print(f"{name}: years present = {years}")
    if incomplete.empty:
        print(f" every present year has all {N_STATES} states")
    else:
        print(f" years without all {N_STATES} states: \n{incomplete}")
    print(
        f" duplicate state-year rows: "
        f"{df.duplicated(['state', 'year']).sum()}"
    )


def report_missing(df: pd.DataFrame, name: str) -> None:
    """
    Print shape, missing-value table and coverage for one table.
    """
    print(f"\n=== {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    summary = missing_summary(df)
    print(summary.to_string())
    print(f" total missing cells: {int(summary.missing_count.sum())}")
    if {"state", "year"} <= set(df.columns):
        coverage_report(df, name)


def missing_data_eda(
    naloxone_path: str = NALOXONE_DISPENSING_FILE,
    mortality_path: str = MORTALITY_RATES,
) -> None:
    """
    Reports missing data in each raw file, in the raw outer merge
    (before cleaning), and in the final cleaned table.
    """
    naloxone = standardize_keys(load_csv(naloxone_path))
    mortality = standardize_keys(load_csv(mortality_path))

    report_missing(naloxone, "Naloxone dispensing (raw)")
    report_missing(mortality, "Mortality (raw)")
    report_missing(
        merge_datasets(naloxone, mortality),
        "Raw outer merge (before cleaning)",
    )
    report_missing(
        build_merged_data(naloxone_path, mortality_path),
        "Final cleaned and merged",
    )


"""
This section of the code identifies variables of interest
and creates a seven-number-summary.
"""


def seven_number_summary(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """
    Mean, std, min, Q1, median, Q3 and max for each column.
    """
    desc = df[columns].describe().T
    desc = desc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    return desc.rename(columns={"25%": "Q1", "50%": "median", "75%": "Q3"})


def categorical_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Every unique value of a column with its count.
    """
    counts = df[column].value_counts().sort_index()
    return counts.rename_axis(column).reset_index(name="count")


def print_categorical(df: pd.DataFrame, column: str, name: str) -> None:
    """
    Prints the unique values and counts for one categorical column.
    Years are listed in full; states are shown compactly because there
    are 51 of them (the full list goes to a CSV file).
    """
    table = categorical_summary(df, column)
    print(f"\n{name} - {column}: {len(table)} unique values")
    if column == "year":
        print(table.to_string(index=False))
    else:
        low, high = int(table["count"].min()), int(table["count"].max())
        print(f"  rows per value: min {low}, max {high}")
        print(f"  values: {', '.join(table[column].astype(str))}")


def summarize_variables(merged: pd.DataFrame) -> None:
    """
    Prints and save summaries of the variables of interest in the
    merged dataset.
    """
    datasets: list[tuple[str, pd.DataFrame, list[str]]] = [
        ("Merged dataset", merged, [NALOXONE, DEATH, PREV])
    ]

    quantitative: list[pd.DataFrame] = []
    categorical: list[pd.DataFrame] = []
    for name, df, columns in datasets:
        print(f"\n=== {name}: {df.shape[0]} rows x {df.shape[1]} columns")
        summary = seven_number_summary(df, columns).round(2)
        print(summary.to_string())
        quantitative.append(summary.assign(dataset=name))
        for column in ["state", "year"]:
            print_categorical(df, column, name)
            table = categorical_summary(df, column)
            table = table.rename(columns={column: "value"})
            categorical.append(table.assign(variable=column, dataset=name))

    pd.concat(quantitative).to_csv("summary_quantitative.csv")
    pd.concat(categorical).to_csv("summary_categorical.csv", index=False)
    print("\nSaved summary_quantitative.csv and summary_categorical.csv")


"""
This section draws the visualizations for the variables of interest.
"""


def savefig(fig: Figure, filename: str) -> None:
    """Tighten layout, save figure at 150 DPI, and close it."""
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

def yearly_averages(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean naloxone dispensing and death rate per year.
    """
    return merged.groupby("year")[[NALOXONE, DEATH]].mean().reset_index()


def fig_yearly_averages(merged: pd.DataFrame) -> None:
    """
    Figure 1: two side-by-side line plots of the average naloxone
    dispensing rate and the average death rate for each year (mean of
    the 51 states). Each panel has its own y-axis starting at zero.
    """
    avg = yearly_averages(merged)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels: list[tuple[Axes, str, str, str, str, str]] = [
        (
            axes[0],
            NALOXONE,
            NALOXONE_LABEL,
            "tab:blue",
            "o",
            "Average naloxone dispensing rate per year",
        ),
        (
            axes[1],
            DEATH,
            DEATH_LABEL,
            "tab:red",
            "s",
            "Average drug overdose death rate per year",
        ),
    ]
    for ax, column, label, color, marker, title in panels:
        ax.plot(
            avg["year"],
            avg[column],
            marker=marker,
            color=color,
            label="Mean of 51 states",
        )
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.set_xticks(avg["year"])
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right")
    savefig(fig, "rq3_fig1.png")


def fig_prev_vs_current(merged: pd.DataFrame, n_labeled: int = 3) -> None:
    """
    Figure 2: previous-year vs. current death rate for all states
    (one dot per state per year) with a no-change line. The highest
    death rates are labeled with their state and year.
    """
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(
        merged[PREV],
        merged[DEATH],
        alpha=0.6,
        label="State-year (all 51 states)",
    )
    top: float = float(max(merged[PREV].max(), merged[DEATH].max()))
    ax.plot(
        [0, top],
        [0, top],
        color="red",
        linestyle="--",
        label="No change from previous year",
    )
    for _, row in merged.nlargest(n_labeled, DEATH).iterrows():
        ax.annotate(
            f"{row['state']} {int(row['year'])}",
            (row[PREV], row[DEATH]),
            textcoords="offset points",
            xytext=(-6, 5),
            ha="right",
            fontsize=9,
        )
    ax.set_title("Previous-year vs. current death rate")
    ax.set_xlabel(PREV_LABEL)
    ax.set_ylabel(DEATH_LABEL)
    ax.legend(loc="upper left")
    savefig(fig, "rq3_fig2.png")


def make_figures(merged: pd.DataFrame) -> None:
    """
    Create and save the figures.
    """
    fig_yearly_averages(merged)
    print("Saved rq3_fig1.png")
    fig_prev_vs_current(merged)
    print("Saved rq3_fig2.png")


def main() -> None:
    """
    Run the full EDA pipeline: build dataset, check missingness,
    summarize variables, compute correlations, and save figures.
    """
    merged = build_merged_data()
    check_merged(merged)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}: {merged.shape}")

    missing_data_eda()

    summarize_variables(merged)

    print("\nCorrelations (merged):")
    print(merged[[NALOXONE, DEATH, PREV]].corr().round(3).to_string())

    make_figures(merged)


if __name__ == "__main__":
    main()
