# brfss_diabetes/preprocessing.py

import pandas as pd
from brfss_diabetes.config import AGE_CATEGORY_MIDPOINTS


def recode_missing(df, column, missing_codes):
    """
    Replace any value in missing codes with pd.NA in df[column].

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to modify.
        missing_codes (iterable): Values to treat as missing, e.g., [7, 8, 9].

    Returns:
        pd.DataFrame: The modified DataFrame with pd.NA in place of missing values.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame columns: {list(df.columns)}"
        )

    if not isinstance(column, str):
        raise ValueError(f"`column` must be a string, got {type(column)}")

    if not isinstance(missing_codes, (list, set, tuple)):
        raise ValueError(
            f"`missing_codes` must be a list, set, or tuple, got {type(missing_codes)}"
        )

    missing_codes = set(missing_codes)

    df[column] = df[column].replace(missing_codes, pd.NA)

    return df


def recode_binary(df, column, yes_codes=[1], no_codes=[2]):
    """
    Recode values into 'Yes', 'No', or pd.NA.

    Parameters:
        df: DataFrame
        column: str — column to recode
        yes_codes: iterable — values to treat as 'Yes'
        no_codes: iterable — values to treat as 'No'

    Returns:
        DataFrame with recoded column (same name, values: 'Yes', 'No', or pd.NA)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame columns: {list(df.columns)}"
        )

    if not isinstance(column, str):
        raise ValueError(f"`column` must be a string, got {type(column)}")

    if not isinstance(yes_codes, (list, set, tuple)):
        raise ValueError(
            f"`yes_codes` must be a list, set, or tuple, got {type(yes_codes)}"
        )

    if not isinstance(no_codes, (list, set, tuple)):
        raise ValueError(
            f"`no_codes` must be a list, set, or tuple, got {type(no_codes)}"
        )

    overlap = set(yes_codes) & set(no_codes)
    if overlap:
        raise ValueError(
            f"`yes_codes` and `no_codes` have overlapping values: {overlap}"
        )

    yes_codes = set(yes_codes)
    no_codes = set(no_codes)

    def map_value(x):
        if pd.isna(x):
            return pd.NA
        elif x in yes_codes:
            return "Yes"
        elif x in no_codes:
            return "No"
        else:
            return pd.NA

    df[column] = df[column].apply(map_value).astype("category")

    return df


def normalize_numeric(df, column, new_column):
    """
    Convert age category codes to midpoint values.
    Category:
        1 -> 18-24 -> 22
        2 -> 25-29 -> 27
        ...
        13 -> 80+   -> 85

    Parameters:
        df: DataFrame
        column: str — column to recode
        new_name: what to call the column after recoding

    Returns:
        DataFrame with recoded column (new_name, values: 22 or 37 or pd.NaN)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame columns: {list(df.columns)}"
        )

    if not isinstance(column, str):
        raise ValueError(f"`column` must be a string, got {type(column)}")

    if not isinstance(new_column, str):
        raise ValueError(f"`new_column` must be a string, got {type(new_column)}")

    df[new_column] = df[column].map(AGE_CATEGORY_MIDPOINTS).astype("Float64")

    return df


def convert_implied_decimal(df, column="_BMI", new_column="BMI"):
    """
    Convert an implied two-place decimal to a decimal (e.g., 3047 -> 30.47)
    Treat any value >= 9000 as missing (pd.NA).

    Parameters:
        df: DataFrame
        column: str — column whose values to convert
        new_name: what to call the column after converting

    Returns:
        DataFrame with recoded column (new_name, value: 33.24, where the old value as 3324)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame columns: {list(df.columns)}"
        )

    if not isinstance(column, str):
        raise ValueError(f"`column` must be a string, got {type(column)}")

    if not isinstance(new_column, str):
        raise ValueError(f"`new_column` must be a string, got {type(new_column)}")

    # 1. Convert to numeric first, safely
    df[column] = pd.to_numeric(df[column], errors="coerce")

    # 2. Remove invalid values (e.g., CDC often codes 9999 as missing)
    df.loc[df[column] >= 9000, column] = pd.NA

    # 3. Scale to decimal
    df[new_column] = df[column] / 100.0
    df[new_column] = df[new_column].astype("Float64")

    # 4. Kill off corrupted or impossible float values
    df.loc[(df[new_column] < 1e-5) | (df[new_column] > 99), new_column] = pd.NA

    return df


def recode_bmi_category(df, column="_BMI5CAT", new_column="BMI_CAT"):
    """
    Convert from a numeric category to a string category for easy of use.
    Map _BMI5CAT codes to text labels:
        1 -> "Underweight"
        2 -> "Normal"
        3 -> "Overweight"
        4 -> "Obese"
        Blank or 9999 -> pd.NA

    Parameters:
        df: DataFrame
        column: str — column whose values to convert
        new_column: what to call the column after converting

    Returns:
        DataFrame with recoded column (new_name, value: Normal, where the old value as 2)
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame columns: {list(df.columns)}"
        )

    if not isinstance(column, str):
        raise ValueError(f"`column` must be a string, got {type(column)}")

    if not isinstance(new_column, str):
        raise ValueError(f"`new_column` must be a string, got {type(new_column)}")

    bmi_mapping = {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}

    df[new_column] = df[column].map(bmi_mapping).astype("category")

    return df


def move_column_to_end(df, column):
    """
    Moves the specified column to the end of the DataFrame if it exists.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to move.

    Returns:
        pd.DataFrame: DataFrame with the specified column at the end.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if not isinstance(column, str):
        raise ValueError(f"`column` must be a string, got {type(column)}")

    if column not in df.columns:
        return df
    cols = [col for col in df.columns if col != column] + [column]
    return df[cols]
