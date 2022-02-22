import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    get_incidence_graph,
    get_structural_incidence_matrix,
)

import pygraphviz as pgv
import networkx.drawing.nx_pylab as nxpl
import networkx.drawing.layout as nx_layout
import matplotlib.pyplot as plt

import scipy.sparse as sps


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


def generate_preliminary_images(show=True, save=False, transparent=True):
    """
    Bipartite graph and incidence matrix

    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()

    m.eq1 = pyo.Constraint(expr=m.x**2 + m.y**2 + m.z**2 == 1)
    m.eq2 = pyo.Constraint(expr=m.x + 2*m.y == 3)

    variables = [m.x, m.y, m.z]
    constraints = [m.eq1, m.eq2]
    graph = get_incidence_graph(variables, constraints)
    n_nodes = len(graph)
    color_map = [
        #"blue" if graph.nodes[i]["bipartite"] == 0 else "orange"
        "blue"
        for i in range(n_nodes)
    ]
    nodes_0 = [n for n in graph if graph.nodes[n]["bipartite"] == 0]
    nodes_1 = [n for n in graph if graph.nodes[n]["bipartite"] == 1]
    pos = nx_layout.bipartite_layout(graph, nodes_0)
    fig = plt.figure()
    nxpl.draw(graph, pos=pos, node_color=color_map, node_size=500, width=2)
    if save:
        plt.savefig("init_bipartite_graph.png", transparent=transparent)
    if show:
        plt.show()

    matrix = get_structural_incidence_matrix(variables, constraints)
    fig = plt.figure()
    plt.spy(matrix, markersize=50)
    if save:
        plt.savefig("init_incidence_matrix.png", transparent=transparent)
    if show:
        plt.show()


def generate_unmatched_variable_images(show=True, save=False):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()

    m.eq1 = pyo.Constraint(expr=m.x**2 + m.y**2 + m.z**2 == 1)
    m.eq2 = pyo.Constraint(expr=m.x + 2*m.y == 3)

    variables = [m.x, m.y, m.z]
    constraints = [m.eq1, m.eq2]
    var_idx_map = ComponentMap((var, i) for i, var in enumerate(variables))
    con_idx_map = ComponentMap((con, i) for i, con in enumerate(constraints))
    graph = get_incidence_graph(variables, constraints)

    n_nodes = len(graph)
    color_map = [
        #"blue" if graph.nodes[i]["bipartite"] == 0 else "orange"
        "blue"
        for i in range(n_nodes)
    ]
    nodes_0 = [n for n in graph if graph.nodes[n]["bipartite"] == 0]
    nodes_1 = [n for n in graph if graph.nodes[n]["bipartite"] == 1]
    pos = nx_layout.bipartite_layout(graph, nodes_0)
    fig = plt.figure()
    nxpl.draw(graph, pos=pos, node_color=color_map, node_size=500, width=2)
    if save:
        plt.savefig("init_bipartite_graph.png", transparent=transparent)
    if show:
        plt.show()

    matrix = get_structural_incidence_matrix(variables, constraints)
    fig = plt.figure()
    plt.spy(matrix, markersize=50)

    # Plot over incidence matrix with matched nodes in a different color
    if save:
        plt.savefig("init_incidence_matrix.png", transparent=transparent)
    if show:
        plt.show()


def generate_unmatched_constraint_images(show=True, save=False):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()

    m.eq1 = pyo.Constraint(expr=m.x**2 + m.y**2 == 1)
    m.eq2 = pyo.Constraint(expr=m.x + 2*m.y == 3)
    m.eq3 = pyo.Constraint(expr=m.x*m.y == 7)


if __name__ == "__main__":
    generate_preliminary_images()
