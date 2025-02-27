from setuptools import setup, find_packages

setup(
    name="electricity_bill_prediction",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "tensorflow",
        "streamlit",
        "flask",
        "joblib"
    ],
    author="Your Name",
    description="A machine learning project to predict electricity bills",
)
