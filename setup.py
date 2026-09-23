from setuptools import setup, find_packages

setup(
    name='thermal-power-cost-prototype',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24,<3',
        'pandas>=2.0,<3',
        'streamlit>=1.50,<2',
        'plotly>=5.18,<7',
        'joblib>=1.3,<2',
        'lightgbm>=4,<5',
        'scikit-learn>=1.2,<2'
    ]
)
