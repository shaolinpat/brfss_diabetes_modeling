# brfss_diabetes/cleaning.py

import pandas as pd
from .preprocessing import (
    recode_missing,
    recode_binary,
    normalize_numeric,
    convert_implied_decimal,
    recode_bmi_category,
)


def clean_common_fields(df: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    # Diabetes uses 7/8/9 for missing
    df = recode_missing(df, "DIABETE4", [7, 8, 9])
    df = recode_binary(df, "DIABETE4", yes_codes=[1, 2], no_codes=[3, 4])

    # _AGEG5YR uses 14 for don't know/refused/missing
    df = recode_missing(df, "_AGEG5YR", [14])
    df = normalize_numeric(df, "_AGEG5YR", "AGE")

    # SEXVAR has no missing code
    sex_map = {1: "Male", 2: "Female"}
    df["SEX"] = df["SEXVAR"].map(sex_map).astype("category")

    # EDUCA uses 9 for missing
    df = recode_missing(df, "EDUCA", [9])
    edu_map = {
        1: "Less than HS",
        2: "Less than HS",
        3: "Less than HS",
        4: "HS or GED",
        5: "Some college",
        6: "College graduate",
    }
    df["EDUCA"] = df["EDUCA"].map(edu_map).astype("category")

    # _BMI5 uses 9 for missing
    df = recode_missing(df, "_BMI5", [9])
    df = convert_implied_decimal(df, column="_BMI5", new_column="BMI")

    # _BMICAT has no missing code
    df = recode_bmi_category(df, column="_BMI5CAT", new_column="BMICAT")

    # SMOKE100 uses 7/9 for missing
    df = recode_missing(df, "SMOKE100", [7, 9])
    df = recode_binary(df, "SMOKE100", yes_codes=[1], no_codes=[2])

    # EXERANY2  uses 7/9 for missing
    df = recode_missing(df, "EXERANY2", [7, 9])
    df = recode_binary(df, "EXERANY2", yes_codes=[1], no_codes=[2])

    return df


def clean_year_specific(df, year):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if not isinstance(year, int):
        raise ValueError(f"`year` must be an int, got {type(df)}")

    if year in [2019, 2020, 2021]:
        df = recode_missing(df, "DRNKANY5", [7, 9])
        df = recode_binary(df, "DRNKANY5", yes_codes=[1], no_codes=[2])
        df["drink_any"] = df["DRNKANY5"]
    else:
        df = recode_missing(df, "DRNKANY6", [7, 9])
        df = recode_binary(df, "DRNKANY6", yes_codes=[1], no_codes=[2])
        df["drink_any"] = df["DRNKANY6"]

    if year in [2019, 2021]:
        df = recode_missing(df, "_FRTLT1A", [9])
        fruit_map = {1: ">= 1x per day", 2: "< 1x per day"}
        df["fruit_low"] = df["_FRTLT1A"].map(fruit_map).astype("category")

        df = convert_implied_decimal(df, "_VEGESU1", "veg_servings")

    if year in [2019, 2022, 2023]:
        df = recode_missing(df, "FOODSTMP", [7, 9])
        df = recode_binary(df, "FOODSTMP", yes_codes=[1], no_codes=[2])
        df["snap_used"] = df["FOODSTMP"]

    if year in [2022, 2023]:
        df = recode_missing(df, "SDHFOOD1", [7, 9])
        food_map = {
            1: "Always",
            2: "Usually",
            3: "Sometimes",
            4: "Rarely",
            5: "Never",
        }
        df["food_insecurity"] = df["SDHFOOD1"].map(food_map).astype("category")

    return df


def clean_brfss(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if not isinstance(year, int):
        raise ValueError(f"`year` must be an int, got {type(df)}")

    df = clean_common_fields(df)
    df = clean_year_specific(df, year)
    df["year"] = year
    return df
