import os
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(proj_root, "src")
sys.path.insert(0, src_path)

import pandas as pd
from utils import clean_brfss, finalize_columns


def main():
    for year in [2019, 2020, 2021, 2022, 2023]:
        # ----------------------------------------------------------------------
        # 1. Define paths
        # ----------------------------------------------------------------------
        in_path = os.path.join(
            os.path.dirname(__file__), f"../data/subset/brfss_subset_{year}.csv"
        )
        out_path = os.path.join(
            os.path.dirname(__file__), f"../data/cleaned/brfss_cleaned_{year}.csv"
        )

        # ----------------------------------------------------------------------
        # 2. Load subset
        # ----------------------------------------------------------------------
        df = pd.read_csv(in_path)

        # ----------------------------------------------------------------------
        # 3. Output the year, columns, shape before
        # ----------------------------------------------------------------------
        df_before = df
        print(f"\nYear: {year}")
        print(f"Columns before: {df_before.columns}")
        print(f"DataFrame shape before: {df_before.shape}")

        # ----------------------------------------------------------------------
        # 4. Clean the data
        # ----------------------------------------------------------------------
        df = clean_brfss(df, year)

        # ---------------------------------------------------------------------------
        # 5. Add a year column
        # ---------------------------------------------------------------------------
        df["year"] = year

        # ---------------------------------------------------------------------------
        # 6. Finalize columns and their order
        # ---------------------------------------------------------------------------

        column_order = [
            "year",
            "AGE",
            "SEX",
            "EDUCA",
            "BMI",
            "BMICAT",
            "drink_any",
            "fruit_low",
            "veg_servings",
            "snap_used",
            "food_insecurity",
            "SMOKE100",
            "EXERANY2",
            "DIABETE4",
        ]
        df = finalize_columns(df, column_order)

        # ----------------------------------------------------------------------
        # 7. Drop missing diabetes status (since it's the target)
        # ----------------------------------------------------------------------
        df = df.dropna(subset=["DIABETE4"])

        # ----------------------------------------------------------------------
        # 8. snake_case the columns
        # ----------------------------------------------------------------------
        df = df.rename(
            columns={
                "AGE": "age",
                "SEX": "sex",
                "EDUCA": "educa",
                "BMI": "bmi",
                "BMICAT": "bmi_cat",
                "SMOKE100": "smoke_100",
                "EXERANY2": "exercise_any",
                "DIABETE4": "diabetes",
            }
        )

        # ----------------------------------------------------------------------
        # 9. Make sure the target column "diabetes" is last
        # ----------------------------------------------------------------------
        target = "diabetes"
        if target in df.columns:
            cols = [col for col in df.columns if col != target] + [target]
            df = df[cols]

        # ----------------------------------------------------------------------
        # 10. Output the columns, shape, number of rows lost
        # ----------------------------------------------------------------------
        print(f"Columns after: {df.columns}")
        print(f"DataFrame shape after: {df.shape}")
        print(
            f"Dropped {df_before.shape[0] - df.shape[0]} rows with missing diabetes status."
        )

        # ----------------------------------------------------------------------
        # 111. Save the cleaned DataFrame
        # ----------------------------------------------------------------------
        df.to_csv(out_path, index=False)
        print(f"Saved clean data to {out_path}")


if __name__ == "__main__":
    main()
