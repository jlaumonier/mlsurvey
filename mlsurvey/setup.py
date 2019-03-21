from setuptools import setup

setup(
    name='mlsurvey',
    version='0.0.1',
    description='Survey of machine learning algorithms',
    author='Julien Laumonier',
    author_email='julien.laumonier@ift.ulaval.ca',
    packages=['mlsurvey'],
    install_requires=['scikit-learn',
                      'numpy',
                      'matplotlib',
                      'joblib',
                      'bokeh',
                      'json2html']
)
