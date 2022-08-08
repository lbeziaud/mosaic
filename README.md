# siMulating sOcioeconomic baSed AffirmatIve aCtion

Louis Béziaud, 2022

> Reardon, S.F., Baker, R., Kasman, M., Klasik, D. and Townsend, J.B. (2018), What Levels of Racial Diversity Can Be Achieved with Socioeconomic-Based Affirmative Action? Evidence from a Simulation Model. J. Pol. Anal. Manage., 37: 630-657. https://doi.org/10.1002/pam.22056

See https://github.com/lbeziaud/re-reardon2018 for details on the replication.

## Content

The main model is [model.py](model.py). Figures are generated by [figures.ipynb](figures.ipynb). The file [utils_figs.py](utils_figs.py) contains auxiliary functions used for launching the experiments. 

## Requirements

This code has been written with Python 3.8. Required packages are listed in [requirements.txt](requirements.txt)

## Usage

1. Clone the repository

```console
$ git clone https://github.com/lbeziaud/mosaic
$ cd mosaic
```

2. Install dependencies (in a virtualenv)

```console
$ python3.8 -m venv venv  # create venv
$ source venv/bin/activate  # activate venv
(venv) $ pip install -r requirements.txt
```

3. Generate the figures (run notebook in headless mode)

```console
(venv) $ jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute figures.ipynb
```

4. Running the model is done by calling `model.run`, which returns a tuple of dataframes:

```python
from model import run

colleges, students, outcomes = run()

colleges.to_csv("colleges.csv")
students.to_csv("students.csv")
outcomes.to_csv("outcomes.csv")
# See also https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
```

Several examples are provided in [utils_figs.py](utils_figs.py) on how to cache the run and on how to run and aggregate repeated experiments.
