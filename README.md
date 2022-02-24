## Incidence Examples
This is a small repository with some examples of the
`incidence_analysis` contributed package in Pyomo.
The `incidence_analysis` code can be found
[here](https://github.com/pyomo/pyomo/tree/main/pyomo/contrib/incidence_analysis).

`incidence_analysis` implements (using NetworkX wherever possible)
three algorithms that are useful for debugging models equation-oriented
chemical process models:
- Maximum matching
- Block triangularization
- Dulmage-Mendelsohn partition

### Contents
This repo has three main parts:
- "example1", an example using the Dulmage-Mendelsohn partition to debug
degrees of freedom in a chemical looping model from IDAES
- "example2", an example using a block triangularization (strongly
connected components) to solve a subsystem of the same
chemical looping model for initialization
- "tutorial", a walkthrough of the three graph algorithms implemented
applied to a small system representing the thermodynamics of a porous
solid particle

### Dependencies
These examples and tutorial depend on the following Python packages:
`idaes-pse`, `pyomo`, `scipy`, `networkx`, `matplotlib`.
I am not 100% sure this is an exhaustive list of dependencies.
These packages (and any others I've forgotten) should be easily
installable with pip.

### Instructions
To run these examples and tutorial, you need to download the code
in this repository, probably by running:

`git clone https://github.com/robbybp/incidence_examples.git`

You then need to "install" this code by running:

`python setup.py develop`

You can check that you have the required dependencies and have
"installed" this code properly by running `pytest` (this may require
a `pip install pytest` if you don't have `pytest` already, although
`pytest` is not the only test runner you could use here).

The examples and tutorial are provided as scripts as well as Jupyter
notebooks (`.ipynb` files).
To open a Jupyter notebook (say to follow along during a demonstration),
navigate to the directory of the notebook you want to open and run:

`jupyter notebook filename.ipynb`
