# tests/test_preprocessing.py

import pandas as pd
import pandas.testing as pdt
import pytest
import brfss_diabetes
from brfss_diabetes import preprocessing as pp

print(brfss_diabetes.__file__)


# ------------------------------------------------------------------------------
# testing def recode_missing(df, column, missing_codes)
# ------------------------------------------------------------------------------


def test_recode_missing():
    df = pd.DataFrame({"col": [1, 7, 8, 9, 2]})
    df = pp.recode_missing(df, "col", [7, 8, 9])
    assert df["col"].isna().sum() == 3
    assert df["col"].iloc[0] == 1
    assert df["col"].iloc[4] == 2


def test_recode_missing_raises_on_not_dataframe():
    # new_name only needed for recode_missing
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        pp.recode_missing("not_a_df", column="age_code", missing_codes=[7])


def test_recode_missing_raises_on_column_missing():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found in DataFrame"):
        pp.recode_missing(df, column="age_code", missing_codes=[7])


def test_recode_missing_raises_on_column_wrong_type():
    df = pd.DataFrame({4: [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`column` must be a string, got <class 'int'>"
    ):
        pp.recode_missing(df, column=4, missing_codes=[7])


def test_recode_missing_replaces_codes():
    df = pd.DataFrame({"col": [1, 7, 9, 2]})
    result = pp.recode_missing(df, "col", [7, 9])
    expected = pd.Series([1, pd.NA, pd.NA, 2], name="col", dtype="object")
    pdt.assert_series_equal(result["col"], expected, check_dtype=False)


def test_recode_missing_invalid_missing_codes_type():
    df = pd.DataFrame({"col": [1, 7, 9]})
    with pytest.raises(
        ValueError, match="`missing_codes` must be a list, set, or tuple"
    ):
        pp.recode_missing(df, "col", 7)  # not a list or set


def test_recode_missing_with_unmatched_code():
    df = pd.DataFrame({"col": [1, 2, 3]})
    result = pp.recode_missing(df, "col", [7, 9])  # 7/9 not in col
    expected = pd.Series([1, 2, 3], name="col")
    pdt.assert_series_equal(result["col"], expected)


# ------------------------------------------------------------------------------
# testing def recode_binary(df, column, yes_codes=[1], no_codes=[2])
# ------------------------------------------------------------------------------


def test_recode_binary():
    df = pd.DataFrame({"col": [1, 2, 3, None]})
    df = pp.recode_binary(df, "col", yes_codes=[1], no_codes=[2])
    expected = pd.Series(["Yes", "No", pd.NA, pd.NA], name="col", dtype="category")
    pdt.assert_series_equal(df["col"], expected, check_dtype=False)


def test_recode_binary_raises_on_not_dataframe():
    # new_name only needed for recode_binary
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        pp.recode_binary("not_a_df", column="age_code", yes_codes=[1], no_codes=[2])


def test_recode_binary_raises_on_column_wrong_type():
    df = pd.DataFrame({4: [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`column` must be a string, got <class 'int'>"
    ):
        pp.recode_binary(df, column=4, yes_codes=[1], no_codes=[2])


def test_recode_binary_raises_on_column_missing():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found in DataFrame"):
        pp.recode_binary(df, column="age_code", yes_codes=[1], no_codes=[2])


def test_recode_binary_replaces_codes():
    df = pd.DataFrame({"col": [1, 2, 3, 4]})
    result = pp.recode_binary(df, "col", [1, 2], [3, 4])
    expected = pd.Series(
        pd.Categorical(["Yes", "Yes", "No", "No"], categories=["No", "Yes"]),
        name="col",
    )
    pdt.assert_series_equal(result["col"], expected, check_dtype=False)


def test_recode_binary_with_unmatched_code():
    df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    result = pp.recode_binary(df, "col", [1, 2], [3, 4])
    expected = pd.Series(
        pd.Categorical(["Yes", "Yes", "No", "No", pd.NA], categories=["No", "Yes"]),
        name="col",
    )
    pdt.assert_series_equal(result["col"], expected)


def test_recode_binary_raises_on_overlapping_codes():
    df = pd.DataFrame({"col": [1, 2]})
    with pytest.raises(ValueError, match="overlapping"):
        pp.recode_binary(df, "col", yes_codes=[1, 2], no_codes=[2, 3])


def test_recode_binary_invalid_yes_code_type():
    df = pd.DataFrame({"col": [1, 2]})
    with pytest.raises(ValueError):
        pp.recode_binary(df, "col", yes_codes=1, no_codes=[2])  # yes_codes not iterable


def test_recode_binary_invalid_no_code_type():
    df = pd.DataFrame({"col": [1, 2]})
    with pytest.raises(ValueError):
        pp.recode_binary(df, "col", yes_codes=[1], no_codes=2)  # no_codes not iterable


# ------------------------------------------------------------------------------
# testing def normalize_numeric(df, column, new_column)
# ------------------------------------------------------------------------------


def test_normalize_numeric():
    df = pd.DataFrame({"age_code": [1, 5, 13, 999]})
    df = pp.normalize_numeric(df, "age_code", "age_midpoint")
    expected = pd.Series(
        [22.0, 42.0, 85.0, pd.NA], name="age_midpoint", dtype="Float64"
    )
    pdt.assert_series_equal(df["age_midpoint"], expected)


def test_normalize_numeric_raises_on_not_dataframe():
    # new_name only needed for normalize_numeric
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        pp.normalize_numeric("not_a_df", column="age_code", new_column="age")


def test_normalize_numeric_raises_on_column_missing():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found in DataFrame"):
        pp.normalize_numeric(df, column="age_code", new_column="age")


def test_normalize_numeric_raises_on_column_wrong_type():
    df = pd.DataFrame({4: [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`column` must be a string, got <class 'int'>"
    ):
        pp.normalize_numeric(df, column=4, new_column="new_col")


def test_normalize_numeric_raises_on_new_column_wrong_type():
    df = pd.DataFrame({"age_code": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`new_column` must be a string, got <class 'int'>"
    ):
        pp.normalize_numeric(df, column="age_code", new_column=7)


# ------------------------------------------------------------------------------
# testing def convert_implied_decimal(df, column="_BMI", new_column="BMI")
# ------------------------------------------------------------------------------


def test_convert_implied_decimal():
    df = pd.DataFrame({"_BMI5": [2550, 9999, 3000, None]})
    df = pp.convert_implied_decimal(df, column="_BMI5", new_column="BMI")
    expected = pd.Series([25.5, pd.NA, 30.0, pd.NA], name="BMI", dtype="Float64")
    pdt.assert_series_equal(df["BMI"], expected)


def test_convert_implied_decimal_raises_on_not_dataframe():
    # new_name only needed for convert_implied_decimal
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        pp.convert_implied_decimal("not_a_df", column="_bmi", new_column="bmi")


def test_convert_implied_decimal_raises_on_column_missing():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found in DataFrame"):
        pp.convert_implied_decimal(df, column="_bmi", new_column="bmi")


def test_convert_implied_decimal_raises_on_column_wrong_type():
    df = pd.DataFrame({4: [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`column` must be a string, got <class 'int'>"
    ):
        pp.convert_implied_decimal(df, column=4, new_column="new_col")


def test_convert_implied_decimal_raises_on_new_column_wrong_type():
    df = pd.DataFrame({"_bmi": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`new_column` must be a string, got <class 'int'>"
    ):
        pp.convert_implied_decimal(df, column="_bmi", new_column=7)


# ------------------------------------------------------------------------------
# testing recode_bmi_category(df, column="_BMI5CAT", new_column="BMI_CAT")
# ------------------------------------------------------------------------------


def test_recode_bmi_category():
    df = pd.DataFrame({"_BMI5CAT": [1, 2, 3, 4, 5, 99]})
    df = pp.recode_bmi_category(df, "_BMI5CAT", "BMI_CAT")
    expected = ["Underweight", "Normal", "Overweight", "Obese", pd.NA, pd.NA]
    assert df["BMI_CAT"].tolist()[:4] == expected[:4]
    assert df["BMI_CAT"].isna().sum() == 2


def test_recode_bmi_category_raises_on_not_dataframe():
    # new_name only needed for recode_bmi_category
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        pp.recode_bmi_category("not_a_df", column="_BMI5CAT", new_column="BMI_CAT")


def test_recode_bmi_category_raises_on_column_missing():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found in DataFrame"):
        pp.recode_bmi_category(df, column="_BMI5CAT", new_column="BMI_CAT")


def test_recode_bmi_category_raises_on_column_wrong_type():
    df = pd.DataFrame({4: [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`column` must be a string, got <class 'int'>"
    ):
        pp.recode_bmi_category(df, column=4, new_column="BMI_CAT")


def test_recode_bmi_category_raises_on_new_column_wrong_type():
    df = pd.DataFrame({"_BMI5CAT": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="`new_column` must be a string, got <class 'int'>"
    ):
        pp.recode_bmi_category(df, column="_BMI5CAT", new_column=7)


# ------------------------------------------------------------------------------
# testing def move_column_to_end(df, column)
# ------------------------------------------------------------------------------
def test_move_column_to_end_moves_column():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = pp.move_column_to_end(df, "b")
    assert list(result.columns) == ["a", "c", "b"]


def test_move_column_to_end_ignores_missing_column():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = pp.move_column_to_end(df, "z")
    assert list(result.columns) == ["a", "b"]


def test_move_column_to_end_raises_on_non_dataframe():
    with pytest.raises(ValueError, match="`df` must be a pandas DataFrame"):
        pp.move_column_to_end("not_a_df", "a")


def test_move_column_to_end_raises_on_non_string_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="`column` must be a string"):
        pp.move_column_to_end(df, 5)
