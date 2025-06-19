# setup.py

from setuptools import setup, find_packages

setup(
    name="flower_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas",
        "scikit-learn",
        "streamlit",
        "shap",
        # …any other deps you need
    ],
)
