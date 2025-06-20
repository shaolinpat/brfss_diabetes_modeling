# setup.py

from setuptools import setup, find_packages

setup(
    name="brfss_diabetes",  # ← renamed
    version="0.1.0",
    packages=find_packages(include=["brfss_diabetes", "brfss_diabetes.*"]),
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "shap",
        "numpy",
    ],
)

