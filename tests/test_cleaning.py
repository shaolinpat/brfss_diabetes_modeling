import pandas as pd
import pytest
from brfss_diabetes.cleaning import (
    clean_common_fields,
    clean_year_specific,
    clean_brfss,
)

# -------------------------------------------------------------------------------
# Test clean_common_fields
# -------------------------------------------------------------------------------


def test_clean_common_fields_minimal_input():
    df = pd.DataFrame(
        {
            "DIABETE4": [1, 3, 7],
            "_AGEG5YR": [1, 2, 14],
            "SEXVAR": [1, 2, 1],
            "EDUCA": [1, 4, 9],
            "_BMI5": [2300, 9999, 1800],
            "_BMI5CAT": [1, 4, 2],
            "SMOKE100": [1, 2, 7],
            "EXERANY2": [1, 2, 9],
        }
    )
    cleaned = clean_common_fields(df.copy())
    assert "AGE" in cleaned
    assert "SEX" in cleaned
    assert "BMI" in cleaned
    assert "BMICAT" in cleaned
    assert cleaned["SEX"].dtype.name == "category"
    assert cleaned["EDUCA"].dtype.name == "category"


def test_clean_common_fields_raises_on_non_dataframe():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        clean_common_fields("not_a_df")


# -------------------------------------------------------------------------------
# Test clean_year_specific
# -------------------------------------------------------------------------------


def test_clean_year_specific_2019_fields():
    df = pd.DataFrame(
        {
            "DRNKANY5": [1, 2, 9],
            "_FRTLT1A": [1, 2, 1],
            "_VEGESU1": [2550, 9999, 3000],
            "FOODSTMP": [1, 2, 9],
        }
    )
    result = clean_year_specific(df.copy(), 2019)
    assert "drink_any" in result
    assert "fruit_low" in result
    assert "veg_servings" in result
    assert result["fruit_low"].dtype.name == "category"


def test_clean_year_specific_raises_on_bad_inputs():
    df = pd.DataFrame({"DRNKANY5": [1]})
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        clean_year_specific("not_a_df", 2019)
    with pytest.raises(ValueError, match="`year` must be an int"):
        clean_year_specific(df, "2019")


def test_clean_year_specific_2023_fields():
    df = pd.DataFrame(
        {"DRNKANY6": [1, 2, 9], "FOODSTMP": [1, 2, 9], "SDHFOOD1": [1, 3, 5]}
    )
    result = clean_year_specific(df.copy(), 2023)

    # Check derived column
    assert "drink_any" in result
    assert "snap_used" in result
    assert "food_insecurity" in result

    # Check types and categories
    assert result["food_insecurity"].dtype.name == "category"
    expected_categories = ["Always", "Usually", "Sometimes", "Rarely", "Never"]
    assert set(result["food_insecurity"].dropna().unique()).issubset(
        set(expected_categories)
    )


# -------------------------------------------------------------------------------
# Test clean_brfss
# -------------------------------------------------------------------------------


def test_clean_brfss_end_to_end():
    df = pd.DataFrame(
        {
            "DIABETE4": [1, 2, 7],
            "_AGEG5YR": [1, 2, 14],
            "SEXVAR": [1, 2, 1],
            "EDUCA": [3, 5, 9],
            "_BMI5": [2200, 9999, 1800],
            "_BMI5CAT": [2, 3, 1],
            "SMOKE100": [1, 2, 7],
            "EXERANY2": [1, 2, 9],
            "DRNKANY5": [1, 2, 9],
            "_FRTLT1A": [1, 2, 1],
            "_VEGESU1": [1500, 9000, 3000],
            "FOODSTMP": [1, 2, 9],
        }
    )

    result = clean_brfss(df.copy(), 2019)
    assert "year" in result
    assert "AGE" in result
    assert "drink_any" in result
    assert "veg_servings" in result
    assert isinstance(result["SEX"].dtype, pd.CategoricalDtype)


def test_clean_brfss_raises_on_not_dataframe():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        clean_brfss("not_a_df", year=2023)


def test_clean_brfss_raises_on_year_not_an_int():
    df = pd.DataFrame(
        {
            "DIABETE4": [1, 2, 7],
            "_AGEG5YR": [1, 2, 14],
            "SEXVAR": [1, 2, 1],
            "EDUCA": [3, 5, 9],
            "_BMI5": [2200, 9999, 1800],
            "_BMI5CAT": [2, 3, 1],
            "SMOKE100": [1, 2, 7],
            "EXERANY2": [1, 2, 9],
            "DRNKANY5": [1, 2, 9],
            "_FRTLT1A": [1, 2, 1],
            "_VEGESU1": [1500, 9000, 3000],
            "FOODSTMP": [1, 2, 9],
        }
    )
    with pytest.raises(ValueError, match="must be an int"):
        clean_brfss(df, year="2023")
