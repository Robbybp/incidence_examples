import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    get_incidence_graph,
    get_structural_incidence_matrix,
)

import networkx.drawing.nx_pylab as nxpl
import networkx.drawing.layout as nx_layout
import matplotlib.pyplot as plt

import scipy.sparse as sps

from incidence_examples.tutorial.model import make_model


def project_onto(coo, rows, cols):
    row_set = set(rows)
    col_set = set(cols)
    new_row = []
    new_col = []
    new_data = []
    for r, c, e in zip(coo.row, coo.col, coo.data):
        if r in row_set and c in col_set:
            new_row.append(r)
            new_col.append(c)
            new_data.append(e)
    return sps.coo_matrix((new_data, (new_row, new_col)), shape=coo.shape)


def generate_dm_images(show=True, save=False, transparent=True):
    m = make_model()
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()

    variables = (
        var_dmp.unmatched
        + var_dmp.underconstrained
        + var_dmp.square
        + var_dmp.overconstrained
    )
    constraints = (
        con_dmp.underconstrained
        + con_dmp.square
        + con_dmp.overconstrained
        + con_dmp.unmatched
    )

    var_idx_map = ComponentMap((var, i) for i, var in enumerate(variables))
    con_idx_map = ComponentMap((con, i) for i, con in enumerate(constraints))

    matrix = get_structural_incidence_matrix(variables, constraints)

    plt.figure()
    markersize = 10
    plt.spy(matrix, markersize=markersize)

    subsystems = []
    colors = []

    underconstrained_vars = var_dmp.unmatched + var_dmp.underconstrained
    underconstrained_cons = con_dmp.underconstrained
    subsystems.append((underconstrained_cons, underconstrained_vars))
    colors.append("orange")

    square_vars = var_dmp.square
    square_cons = con_dmp.square
    subsystems.append((square_cons, square_vars))
    colors.append("green")

    overconstrained_vars = var_dmp.overconstrained
    overconstrained_cons = con_dmp.overconstrained + con_dmp.unmatched
    subsystems.append((overconstrained_cons, overconstrained_vars))
    colors.append("red")

    for i, (constraints, variables) in enumerate(subsystems):
        color = colors[i]
        rows = [con_idx_map[con] for con in constraints]
        cols = [var_idx_map[var] for var in variables]
        proj_matrix = project_onto(matrix, rows, cols)
        plt.spy(proj_matrix, color=color, markersize=markersize)

    if save:
        plt.savefig("dulmage_mendelsohn.png", transparent=transparent)
    if show:
        plt.show()


if __name__ == "__main__":
    generate_dm_images(save=True)
