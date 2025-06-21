# brfss_diabetes/io.py

import pandas as pd
import sys
from pathlib import Path

from .preprocessing import move_column_to_end


def get_csv(year, data_dir=Path("../data/cleaned")):
    """
    Take a year and put the data from a csv for that year and put it into a DataFrame.

    Parameters:
        year: int representation of the year
        data_dir: directory relative to the calling notebook or script.

    Returns:
        pd.DataFrame with the csv data loaded.
    """
    if not isinstance(year, int):
        raise ValueError(f"`year` must be an int, got {type(year)}")

    if not isinstance(data_dir, Path):
        raise ValueError(f"`data_dir` must be a pathlib.Path, got {type(data_dir)}")

    filename = f"brfss_cleaned_{year}.csv"

    if "google.colab" in sys.modules:
        url = f"https://raw.githubusercontent.com/shaolinpat/brfss_diabetes_modeling/main/data/cleaned/{filename}"
        print(f"[Colab] Loading from GitHub: {url}")
        return pd.read_csv(url, low_memory=False)
    else:
        path = Path(data_dir / filename)
        print(f"[Local] Loading from: {path}")
        return pd.read_csv(path, low_memory=False)


def load_all_years(years, data_dir=Path("../data/cleaned")):
    """
    Load and merge cleaned BRFSS CSV files for multiple years.

    Parameters:
        years (list of int): List of years to load.
        data_dir (Path): Directory containing cleaned CSVs.

    Returns:
        pd.DataFrame: Combined DataFrame with 'diabetes' column at end.
    """
    if not isinstance(years, list) or not all(isinstance(y, int) for y in years):
        raise ValueError("`years` must be a list of integers")

    if not isinstance(data_dir, Path):
        raise ValueError(f"`data_dir` must be a pathlib.Path, got {type(data_dir)}")

    dfs = []
    for year in years:
        df = get_csv(year, data_dir=data_dir)

        if "diabetes" not in df.columns:
            raise ValueError(f"'diabetes' column missing in {year}")

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return move_column_to_end(df_all, "diabetes")


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
