# tests/test_preprocessing.py

import pandas as pd
import pandas.testing as pdt
import pytest
from brfss_diabetes import preprocessing as pp


def test_recode_missing():
    df = pd.DataFrame({"col": [1, 7, 8, 9, 2]})
    df = pp.recode_missing(df, "col", [7, 8, 9])
    assert df["col"].isna().sum() == 3
    assert df["col"].iloc[0] == 1
    assert df["col"].iloc[4] == 2


def test_recode_binary():
    df = pd.DataFrame({"col": [1, 2, 3, None]})
    df = pp.recode_binary(df, "col", yes_codes=[1], no_codes=[2])
    expected = pd.Series(["Yes", "No", pd.NA, pd.NA], name="col", dtype="category")
    pdt.assert_series_equal(df["col"], expected, check_dtype=False)


def test_normalize_numeric():
    df = pd.DataFrame({"age_code": [1, 5, 13, 999]})
    df = pp.normalize_numeric(df, "age_code", "age_midpoint")
    expected = pd.Series(
        [22.0, 42.0, 85.0, pd.NA], name="age_midpoint", dtype="Float64"
    )
    pdt.assert_series_equal(df["age_midpoint"], expected)


def test_convert_implied_decimal():
    df = pd.DataFrame({"_BMI5": [2550, 9999, 3000, None]})
    df = pp.convert_implied_decimal(df, column="_BMI5", new_column="BMI")
    expected = pd.Series([25.5, pd.NA, 30.0, pd.NA], name="BMI", dtype="Float64")
    pdt.assert_series_equal(df["BMI"], expected)


#    assert df["BMI"].tolist() == [25.5, pd.NA, 30.0, pd.NA]


def test_recode_bmi_category():
    df = pd.DataFrame({"_BMI5CAT": [1, 2, 3, 4, 5, 99]})
    df = pp.recode_bmi_category(df, "_BMI5CAT", "BMI_CAT")
    expected = ["Underweight", "Normal", "Overweight", "Obese", pd.NA, pd.NA]
    assert df["BMI_CAT"].tolist()[:4] == expected[:4]
    assert df["BMI_CAT"].isna().sum() == 2
