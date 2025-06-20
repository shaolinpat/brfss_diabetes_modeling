# tests/test_io.py

import pandas as pd
import pytest
from brfss_diabetes.io import finalize_columns


def test_finalize_columns_keeps_only_specified_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    result = finalize_columns(df, ["A", "C"])
    expected = pd.DataFrame({"A": [1, 2], "C": [5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_finalize_columns_with_missing_columns_drops_them_silently():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = finalize_columns(df, ["A", "C"])  # C does not exist
    expected = pd.DataFrame({"A": [1, 2]})
    pd.testing.assert_frame_equal(result, expected)


def test_finalize_columns_raises_on_non_dataframe():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        finalize_columns("not_a_df", ["A"])


def test_finalize_columns_raises_on_keep_cols_not_list():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError, match="Keep_cols .* is not a list"):
        finalize_columns(df, "A")
