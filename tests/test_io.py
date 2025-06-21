# tests/test_io.py

import pandas as pd
from pathlib import Path
import pytest
import sys
from unittest.mock import patch, MagicMock

from brfss_diabetes.io import get_csv, load_all_years, finalize_columns


# ------------------------------------------------------------------------------
# testing def get_csv(year, data_dir=Path("../data/cleaned"))
# ------------------------------------------------------------------------------


def test_get_csv_raises_on_non_integer_year():
    with pytest.raises(ValueError, match="`year` must be an int"):
        get_csv("2020")  # string instead of int


def test_get_csv_raises_on_invalid_data_dir_type():
    with pytest.raises(ValueError, match="`data_dir` must be a pathlib.Path"):
        get_csv(2020, data_dir="not_a_path")


def test_get_csv_reads_local_file(monkeypatch):
    monkeypatch.setitem(sys.modules, "not_colab", MagicMock())  # simulate non-Colab

    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = pd.DataFrame({"a": [1, 2]})
        mock_read_csv.return_value = mock_df

        df = get_csv(2020, data_dir=Path("/some/path"))
        assert df.equals(mock_df)
        mock_read_csv.assert_called_once()


def test_get_csv_reads_from_github_in_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", MagicMock())  # simulate Colab

    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = pd.DataFrame({"a": [1, 2]})
        mock_read_csv.return_value = mock_df

        df = get_csv(2021)
        assert df.equals(mock_df)
        mock_read_csv.assert_called_once()
        assert "raw.githubusercontent.com" in mock_read_csv.call_args[0][0]


# ------------------------------------------------------------------------------
# testing def load_all_years(years, data_dir=Path("../data/cleaned"))
# ------------------------------------------------------------------------------
def test_load_all_years_raises_on_years_not_list():
    with pytest.raises(ValueError, match="`years` must be a list of integers"):
        load_all_years("2020")  # string instead of list


def test_load_all_years_raises_on_years_with_non_int():
    with pytest.raises(ValueError, match="`years` must be a list of integers"):
        load_all_years([2020, "2021", 2022])  # one item is a string


def test_load_all_years_on_invalid_data_dir_type():
    with pytest.raises(ValueError, match="`data_dir` must be a pathlib.Path"):
        load_all_years([2020], data_dir="not_a_path")


def test_load_all_years_on_missing_diabetes_column():
    # Simulate a DataFrame without 'diabetes' column
    mock_df = pd.DataFrame({"feature": [1, 2]})

    with patch("brfss_diabetes.io.get_csv", return_value=mock_df):
        with pytest.raises(ValueError, match="'diabetes' column missing in 2020"):
            load_all_years([2020])


def test_load_all_years_reads_and_merges(monkeypatch):
    mock_df = pd.DataFrame({"feature": [1, 2], "diabetes": [0, 1]})

    # Patch get_csv to return mock_df for any year
    with patch("brfss_diabetes.io.get_csv", return_value=mock_df) as mock_get_csv:
        result = load_all_years([2019, 2020, 2021])

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 6  # 2 rows * 3 years
        assert "diabetes" in result.columns
        assert result.columns[-1] == "diabetes"  # diabetes should be last column
        assert mock_get_csv.call_count == 3


# ------------------------------------------------------------------------------
# testing def finalize_columns(df, keep_cols: list[str]) -> pd.DataFrame
# ------------------------------------------------------------------------------


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
