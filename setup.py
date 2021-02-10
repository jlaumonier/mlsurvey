import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlsurvey",  # Replace with your own username
    version="0.0.1",
    author="Julien Laumônier",
    author_email="julien.laumonier@iid.ulaval.ca",
    description="Package used to test the implementation/use of machine learning tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scikit-learn',
                      'numpy',
                      'matplotlib',
                      'joblib',
                      'json2table',
                      'pandas',
                      'dash',
                      'dash_core_components',
                      'dash_dangerously_set_inner_html',
                      'colorlover',
                      'plotly',
                      'tqdm',
                      'tinydb',
                      'liac-arff',
                      'tables',
                      'xlrd',
                      'mlflow',
                      'xlsxwriter',
                      'openpyxl',
                      'kedro'],
    python_requires='>=3.6'
)
