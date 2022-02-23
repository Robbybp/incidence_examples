import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    get_incidence_graph,
    get_structural_incidence_matrix,
)

import networkx as nx
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


def plot_blt_incidence_matrix(show=True, save=False, transparent=True):
    """
    Plot block lower triangular incidence matrix
    """
    m = make_model()

    #
    # Apply fix
    #
    m.particle_porosity.unfix()

    @m.Constraint()
    def flow_density_eqn(b):
        return m.flow_mass == m.velocity * m.area * m.dens_mass_particle
    ###

    igraph = IncidenceGraphInterface(m)
    var_blocks, con_blocks = igraph.get_diagonal_blocks()

    variables = sum(var_blocks, [])
    constraints = sum(con_blocks, [])

    var_idx_map = ComponentMap((var, i) for i, var in enumerate(variables))
    con_idx_map = ComponentMap((con, i) for i, con in enumerate(constraints))

    matrix = get_structural_incidence_matrix(variables, constraints)

    plt.figure()
    markersize = 10
    plt.spy(matrix, markersize=markersize)

    for i, (constraints, variables) in enumerate(zip(con_blocks, var_blocks)):
        color = "orange" if i % 2 else "green"
        #color = "orange"
        rows = [con_idx_map[con] for con in constraints]
        cols = [var_idx_map[var] for var in variables]
        proj_matrix = project_onto(matrix, rows, cols)
        plt.spy(proj_matrix, color=color, markersize=markersize)

    if save:
        plt.savefig("blt.png", transparent=transparent)
    if show:
        plt.show()


def plot_blt_algorithm_steps(show=True, save=False, transparent=True):
    """
    Plot intermediate graphs used in the construction of a block lower
    triangular permutation.
    """
    m = make_model()

    #
    # Apply fix
    #
    m.particle_porosity.unfix()

    @m.Constraint()
    def flow_density_eqn(b):
        return m.flow_mass == m.velocity * m.area * m.dens_mass_particle
    ###

    igraph = IncidenceGraphInterface(m)

    #
    # Step 1: perfect matching
    #
    matching = igraph.maximum_matching()
    constraints = igraph.constraints
    variables = [matching[con] for con in constraints]
    graph = get_incidence_graph(variables, constraints)
    n_nodes = len(graph)
    color_map = ["blue" for i in range(n_nodes)]
    nodes_0 = [n for n in graph if graph.nodes[n]["bipartite"] == 0]
    nodes_1 = [n for n in graph if graph.nodes[n]["bipartite"] == 1]
    pos = nx_layout.bipartite_layout(graph, nodes_0)
    fig = plt.figure()
    nxpl.draw(graph, pos=pos, node_color=color_map, node_size=100, width=2)
    if save:
        plt.savefig("blt_matching.png", transparent=transparent)
    if show:
        plt.show()
    ###

    #
    # Step 2: "Project" into directed graph
    #
    var_idx_map = ComponentMap((var, i) for i, var in enumerate(variables))
    con_idx_map = ComponentMap((con, i) for i, con in enumerate(constraints))
    matching = {
        con_idx_map[con]: var_idx_map[var] for con, var in matching.items()
    }
    dg = nx.DiGraph()
    M = len(constraints)
    dg.add_nodes_from(range(M))
    for n in dg.nodes:
        col_idx = matching[n]
        col_node = col_idx + M
        for neighbor in graph[col_node]:
            if neighbor != n:
                dg.add_edge(neighbor, n)
    plt.figure()
    nxpl.draw(dg, node_size=100, width=2)
    if save:
        plt.savefig("blt_digraph.png", transparent=transparent)
    if show:
        plt.show()
    ###

    #
    # Step 3: Strongly connected components and convert to DAG
    #
    scc_list = list(nx.algorithms.components.strongly_connected_components(dg))
    node_scc_map = {n: idx for idx, scc in enumerate(scc_list) for n in scc}
    dag = nx.DiGraph()
    for i, c in enumerate(scc_list):
        dag.add_node(i)
    for n in dg.nodes:
        source_scc = node_scc_map[n]
        for neighbor in dg[n]:
            target_scc = node_scc_map[neighbor]
            if target_scc != source_scc:
                dag.add_edge(target_scc, source_scc)
    plt.figure()
    nxpl.draw(dag, node_size=100, width=2)
    if save:
        plt.savefig("blt_dag.png", transparent=transparent)
    if show:
        plt.show()
    ###

    #
    # Step 4 is topological sort. I'll just plot the finished product
    #
    var_blocks, con_blocks = igraph.get_diagonal_blocks()
    variables = sum(var_blocks, [])
    constraints = sum(con_blocks, [])
    matrix = get_structural_incidence_matrix(variables, constraints)
    plt.figure()
    plt.spy(matrix, markersize=10)
    if save:
        plt.savefig("blt_matrix.png", transparent=transparent)
    if show:
        plt.show()


if __name__ == "__main__":
    plot_blt_incidence_matrix(save=True)
    plot_blt_algorithm_steps(save=True)
