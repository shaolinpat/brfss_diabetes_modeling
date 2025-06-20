# brfss_diabetes/io.py

import pandas as pd


def finalize_columns(df, keep_cols: list[str]) -> pd.DataFrame:
    """
    Take a dataframe and a list of columns to keep and return the resultind datafram.

    Parameters:
        df: DataFrame
        keep_cols: list of columns to keep

    Returns:
        DataFrame with kept columns as indicated in keep_columns
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`df` must be a pandas DataFrame, got {type(df)}")

    if not isinstance(keep_cols, list):
        raise KeyError(f"Keep_cols '{keep_cols}' is not a list")

    keep = [col for col in keep_cols if col in df.columns]
    extra = [col for col in df.columns if col not in keep_cols]
    if extra:
        print(f"Dropping unused columns: {extra}")

    return df[keep]
