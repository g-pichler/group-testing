# group-testing

This code can be used to explore the results of

```
"Modelling the Utility of Group Testing for Public Health Surveillance,"
Günther Koliander and Georg Pichler, 2020.
```

## Installation

Make sure the [requirements](requirements.txt) are satisfied.
Other than that, no installation is required. The main `pooltesting` package may be installed using [setup.py](setup.py), but this is not necessary.

## Usage

Run a Jupyter Notebook in the main directory and open [pooltesting_plot.ipynb](pooltesting_plot.ipynb). Upon executing all cells, this will present an interactive plot of all fundamentals bounds, the `2SG`, `individual testing`, and `binary splitting` strategies. The `1SG` strategy can be un-commented in the code of [util.py](util.py). The plot can be zoomed and panned. Sliders at the bottom allow changing the values p, a, b, N for each subpopulation.

## Unittests

Rudimentary, randomized unitittests are included in [pooltesting/test_pooltest.py](pooltesting/test_pooltest.py).
They can be executed  using
```
python -m unittest pooltesting.test_pooltest.TestPoolTest
```
and should take several seconds to complete.

## Code
The main code in [pooltesting/pooltest.py](pooltesting/pooltest.py) is documented using Python Docstrings.
