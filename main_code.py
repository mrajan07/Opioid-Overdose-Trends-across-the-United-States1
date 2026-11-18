import pandas as pd

NALOXONE_DISPENSING_FILE = "State Naloxone Dispensing Rates.csv"
MORTALITY_RATES = "Drug Overdose Mortality by State.csv"
OUTPUT_FILE = "merged_data_file.csv"

YEAR_TO_REMOVE = 2017
N_STATES = 51

DROP_COLUMNS = [
    "STATE_FIPS", "STATE_ABBREV", "URL", "Deaths",
    "Naloxone Dispensing Rate (per 100 persons)"]

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

def standardize_keys(df):
    df = df.rename(columns = {"Year": "year", "YEAR": "year",
                              "Location": "state"})
    df["year"] = df["year"].astype(int)
    df["state"] = df["state"].str.strip()
    return df

def merge_datasets(naloxone, mortality):
    merged = mortality.merge(naloxone, on=["state", "year"], how="outer")
    return merged 

def clean_merged(df):
    df = df.drop(columns = DROP_COLUMNS)
    df = df.rename(columns = {"Naloxone Dispensing Rate": "naloxone_dispensing_rate_per_100", 
                              "Death Rate": "death_rate_per_100000"}
                             )
    df = df[["state", "year"] +[c for c in df.columns if c not in ("state", "year")]]
    df = df[df["year"] > YEAR_TO_REMOVE]
    df = df.sort_values(["state", "year"]).reset_index(drop = True)
    return df 
    
def add_previous_year_mortality(df):
    prev = df[["state", "year", "death_rate_per_100000"]].copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns = {
        "death_rate_per_100000":
        "prev_year_death_rate_per_100000"})
    return df.merge(prev, on=["state", "year"], how = "left")

def drop_incomplete_rows(df):
    return df.dropna().reset_index(drop=True)

def check_merged(df):
    assert df.shape[0] == 51*6
    assert df["year"].min() == 2019
    assert df["year"].max() == 2024
    assert not df.duplicated(["state", "year"]).any()
    assert not df.isna().any().any()
    assert (df["naloxone_dispensing_rate_per_100"] >= 0).all()

def build_merged_data(naloxone_path = NALOXONE_DISPENSING_FILE,
                      mortality_path = MORTALITY_RATES):
    
    naloxone = standardize_keys(load_csv(naloxone_path))
    mortality = standardize_keys(load_csv(mortality_path))

    merged = merge_datasets(naloxone, mortality)
    cleaned = clean_merged(merged)
    cleaned = add_previous_year_mortality(cleaned)
    return drop_incomplete_rows(cleaned)

def main():
    merged = build_merged_data()
    check_merged(merged)
    merged.to_csv(OUTPUT_FILE, index = False)
    print(merged.shape)
    print(merged.head(8))
    print(merged.describe().round(2))

if __name__ == "__main__":
    main()


        
            