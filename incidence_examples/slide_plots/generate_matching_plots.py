import pyomo.environ as pyo
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    get_incidence_graph,
    get_structural_incidence_matrix,
)

import pygraphviz as pgv
import networkx.drawing.nx_pylab as nxpl
import networkx.drawing.layout as nx_layout
import matplotlib.pyplot as plt


def generate_preliminary_images(show=True, save=False):
    """
    Bipartite graph and incidence matrix

    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()

    m.eq1 = pyo.Constraint(expr=m.x**2 + m.y**2 + m.z**2 == 1)
    m.eq2 = pyo.Constraint(expr=m.x + 2*m.y == 3)

    
def generate_unmatched_variable_images(show=True, save=False):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()

    m.eq1 = pyo.Constraint(expr=m.x**2 + m.y**2 + m.z**2 == 1)
    m.eq2 = pyo.Constraint(expr=m.x + 2*m.y == 3)


def generate_unmatched_constraint_images(show=True, save=False):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()

    m.eq1 = pyo.Constraint(expr=m.x**2 + m.y**2 == 1)
    m.eq2 = pyo.Constraint(expr=m.x + 2*m.y == 3)
    m.eq3 = pyo.Constraint(expr=m.x*m.y == 7)


if __name__ == "__main__":
    generate_preliminary_images()
