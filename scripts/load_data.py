import pandas as pd
import os


def load_brfss_raw(path_to_raw):
    """
    Load the raw BRFSS 2023 file into a pandas Dataframe.
    path_to_raw: str, full path to the .ASC file
    """
    #
    df = pd.read_sas(path_to_raw, format="xport", encoding="utf-8")
    return df


def subset_variables(df, columns):
    """
    Keep only the columns in the "columns" list.
    """
    return df[columns].copy()


def get_vars_to_keep(year: int) -> list[str]:
    return vars_common + vars_by_year.get(year, [])


if __name__ == "__main__":
    for year in [2019, 2020, 2021, 2022, 2023]:

        # 1. Define paths
        raw_path = os.path.join(
            os.path.dirname(__file__), f"../data/raw/multi_year/{year}/LLCP{year}.XPT"
        )
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"../data/subset/brfss_subset_{year}.csv",
        )

        # 2. List variables to keep that are common to all years
        vars_common = [
            "DIABETE4",
            "_AGEG5YR",
            "SEXVAR",
            "EDUCA",
            "_BMI5",
            "_BMI5CAT",
            "SMOKE100",
            "EXERANY2",
        ]

        # 3. Year-specific variables
        # DRNKANY5 (used 2019–2021), renamed to DRNKANY6 in 2022+
        vars_by_year = {
            2019: ["DRNKANY5", "_FRTLT1A", "_VEGESU1", "FOODSTMP"],
            2020: ["DRNKANY5"],
            2021: ["DRNKANY5", "_FRTLT1A", "_VEGESU1"],
            2022: ["DRNKANY6", "SDHFOOD1", "FOODSTMP"],
            2023: ["DRNKANY6", "SDHFOOD1", "FOODSTMP"],
        }

        # 3. Load raw file
        df_raw = load_brfss_raw(raw_path)

        # 4. Subset
        df_subset = subset_variables(df_raw, get_vars_to_keep(year))

        # Move 'diabetes' column to the end if it's present
        if "DIABETE4" in df_subset.columns:
            cols = [col for col in df_subset.columns if col != "DIABETE4"] + [
                "DIABETE4"
            ]
            df_subset = df_subset[cols]

            print(f"Rewritten with target last: {out_path}")
        else:
            print(f"'DIABETE4' not found in: {out_path}")

        # 5. Save to disk
        df_subset.to_csv(out_path, index=False)
        print(f"Saved subset to {out_path}")
        print(f"Shape of output file:  {df_subset.shape}")
