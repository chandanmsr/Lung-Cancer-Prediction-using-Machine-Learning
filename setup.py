from setuptools import setup, find_packages

setup(
    name="lung_cancer_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'xgboost>=1.5.0',
        'imbalanced-learn>=0.8.0',
        'tabulate>=0.8.0',
    ],
    python_requires='>=3.8',
)