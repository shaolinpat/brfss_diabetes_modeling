import pandas as pd


def recode_missing(df, column, missing_codes):
    """
    Replace any value in missing codes with pd.NA in df[column].
    missing_codes: list of integers(e.g., [7, 8, 9])
    """
    df[column] = df[column].replace(missing_codes, pd.NA)
    return df


def recode_diabetes(df, column="DIABETE4"):
    """
    Map DIABETE4 to text: 1 -> "Yes", 2 -> "No, else -> pd.NA
    """

    # 1. Verify that df is a DataFrame
    if not hasattr(df, "columns"):
        raise ValueError("`df` must be a pandas DataFrame, but got: %r" % df)

    # 2. Verify that the specified column is in df
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    mapping = {1: "Yes", 2: "No", 7: pd.NA, 9: pd.NA}

    # 3. Perform the map and convert to category
    df[column] = df[column].map(mapping).astype("category")

    return df


def recode_binary(df, column, yes_codes=[1], no_codes=2):
    """
    Generic binary recode: yes_codes -> "yes", no_codes = -> "No, else pd.NA
    """
    df[column] = df[column].apply(
        lambda x: (
            pd.NA
            if pd.isna(x)
            else "Yes" if x in yes_codes else "No" if x in no_codes else pd.NA
        )
    )

    return df


def normalize_numeric(df, column, new_name):
    """
    Example: turn an age age group code into the midpoint of that age bin.
    Convert each code to a number.
    """
    age_mapping = {
        1: 22,
        2: 27,
        3: 32,
        4: 37,
        5: 42,
        6: 47,
        7: 52,
        8: 57,
        9: 62,
        10: 67,
        11: 72,
        12: 77,
        13: 85,
    }
    df[new_name] = df[column].map(age_mapping).astype("float")

    return df


def convert_implied_decimal(df, column="_BMI", new_column="BMI"):
    """
    Convert an implied two-place decimal to a decimal (e.g., 3047 -> 30.47)
    Treat any value >= 9000 as missing (pd.NA).
    """
    # 1. Convert to numeric first, safely
    df[column] = pd.to_numeric(df[column], errors="coerce")

    # 2. Remove invalid values (e.g., CDC often codes 9999 as missing)
    df.loc[df[column] >= 9000, column] = pd.NA

    # 3. Scale to decimal
    df[new_column] = df[column] / 100.0

    # 4. Kill off corrupted or impossible float values
    df.loc[(df[new_column] < 1e-5) | (df[new_column] > 99), new_column] = pd.NA

    return df


def recode_bmi_category(df, column="_BMI5CAT", new_column="BMI_CAT"):
    """
    Map _BMI5CAT codes to text labels:
        1 -> "Underweight"
        2 -> "Normal"
        3 -> "Overweight"
        5 -> "Obese"
        Blank or 9999 -> pd.NA
    """
    bmi_mapping = {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}

    df[new_column] = df[column].map(bmi_mapping).astype("category")

    return df


def clean_common_fields(df: pd.DataFrame) -> pd.DataFrame:
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
    df = clean_common_fields(df)
    df = clean_year_specific(df, year)
    df["year"] = year
    return df


def finalize_columns(df, keep_cols: list[str]) -> pd.DataFrame:
    keep = [col for col in keep_cols if col in df.columns]
    extra = [col for col in df.columns if col not in keep_cols]
    if extra:
        print(f"Dropping unused columns: {extra}")
    return df[keep]
