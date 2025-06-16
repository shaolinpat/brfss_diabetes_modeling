# BRFSS Diabetes Modeling

[![CI](https://github.com/shaolinpat/brfss_diabetes_cleaning/actions/workflows/ci.yml/badge.svg)](https://github.com/shaolinpat/brfss_diabetes_cleaning/actions/workflows/ci.yml)
[![Coverage (flag)](https://img.shields.io/codecov/c/github/shaolinpat/brfss_diabetes_cleaning.svg?flag=flower_classifier&branch=main)](https://codecov.io/gh/shaolinpat/brfss_diabetes_cleaning)  
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaolinpat/brfss_diabetes_cleaning/blob/main/notebooks/00_logistic_regression_all_years.ipynb)


An end-to-end Iris-dataset classifier with EDA, 8 models, SHAP interpretation, and a Streamlit UI.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Visual Highlights](#visual-highlights)
- [Features](#features)
- [Quick Start](#quick-start)
- [Launch Streamlit](#launch-the-streamlit-app)
- [Quick Verify](#quick-verify)
- [File Layout](#file-layout)
- [Next Steps](#next-steps)
- [License](#license)

---

## Overview

---

## Visual Highlights


---

## Features 

---
## Quick Start
_(Run each command below at the command line, not inside Python or Jupyter)_

### Run in Google Colab
Try it now with no setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaolinpat/flower_classifier/blob/main/notebooks/flower_classifier.ipynb)


### Run Locally with Conda

### 1. Clone this repo
```bash
git clone git@github.com:shaolinpat/brfss_diabetes_cleaning.git  
cd brfss_diabetes_cleaning  
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate brfss_env
bash scripts/register_kernel.sh
```

### 3. Launch the notebook
```bash
# Option 1: Run the notebooks in your default web browser
jupyter notebook notebooks/00_logistic_regression_all_years.ipynb
jupyter notebook notebooks/01_xgboost_2022_2023.ipynb

# Option 2: Open them directly in VS Code (if installed and in your PATH)
code notebooks/00_logistic_regression_all_years.ipynb
code notebooks/01_xgboost_2022_2023.ipynb


---

## Launch the Streamlit app


---

## Quick Verify


---

## File Layout

---

## Next Steps



2022, 2023
- DIABETE4: "Ever told you have diabetes?" (1=Yes, 2=Yes, but female told only during preganacy, 3=No, 4=No, pre-diabtees or borderline diabetes 7=Don't know/Not Sure, 8=Refused, 9=Missing, BLANK=Not asked or Missing)
- _AGEG5YR: Age in 5-year groups (1=18-24, 2=25-29, ..., 13=80+, 14=Don't know/Refused/Missing)
- SEXVAR: (1=Male, 2=Female)
- EDUCA: Education level (1=Never attended/only K, 2=Grades 1 through 8, 3=Grades 9 through 11, 4=Grade 12 or GED, 5=Some college, 6=College graduate, 9=Refused, BLANK=Not asked or Missing)
- _BMI5: Calculated BMI category (1-9999=1 or greater, BLANK=Don't know/Refused/Missing)
- _BMI5CAT: Computed body mass index categories (1=Underweight, 2=Normal, 3=Overweight, 4=Obese, BLANK=Don't know/Refused/Missing)
- SMOKE100: Smoked at Least 100 Cigarettes (1=Yes, 2=No, 7=Don't know/Not Sure, 9=Refused, BLANK=Not asked or Missing)
- DRNKANY6: Drink any alcoholic beverages in past 30 days (1=Yes, 2=No, 7=Don't know/Not Sure, 9=Refused/Missing)
- EXERANY2: Exercise in past 30 days (1=Yes, 2=No, 7=Don't know/Not Sure, 9=Refused, BLANK=Not asked or Missing)
- SDHFOOD1:How often did the food that you bought not last, and you didn't have money to get more?
 (1=Always, 2=Usually, 3=Sometimes, 4=Rarely, 5=Never, 7=Don't know/Not Sure, 9=Refused, BLANK=Not asked or missing)


2019, 2020, 2021
- _FRTLT1A: Consume Fruit 1 or more times per day (1=Consumed fruit one or more times per day, 2=Consumend fruit < one tim per day, 9=Don't know, refused or missing values)
- _VEGESU1: Total vegetables consumed per day (numeric, BLANK=Not asked or Missing)
- DRNKANY5: Any alcohol consumption in past 30 days (1=Yes, 2=No, 7=Don't know/Not Sure, 9=Refused/Missing)


Note: SDHFOOD1 (added 2022+) measures perceived food insecurity in the past 30 days. 
FOODSTMP (2019–2021) measures SNAP receipt over the past 12 months. 
These are related but not equivalent indicators of food access.


## License

This project is licensed under the [MIT License](LICENSE).


*Built with scikit-learn 1.6 · Streamlit 1.33 · Python 3.11*