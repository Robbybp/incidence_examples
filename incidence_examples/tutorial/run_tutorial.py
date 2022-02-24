import pyomo.environ as pyo
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    get_structural_incidence_matrix,
)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.common.collections import ComponentSet

from idaes.core.util.model_statistics import degrees_of_freedom

import numpy as np
import matplotlib.pyplot as plt

from incidence_examples.tutorial.model import make_model

def main(save=False, show=True, transparent=True):
    m = make_model()

    #
    # Check system for structural nonsingularity
    #
    igraph = IncidenceGraphInterface(m)
    matching = igraph.maximum_matching()
    M = len(igraph.constraints)
    N = len(igraph.variables)
    print("Degrees of freedom = %s" % degrees_of_freedom(m))
    print(M, N, len(matching))
    print("Matching:")
    for con, var in matching.items():
        print()
        print(con.name)
        print(var.name)
    print()

    print("Unmatched constraints:")
    for con in igraph.constraints:
        if con not in matching:
            print("  %s" % con.name)

    print("Unmatched variables:")
    matched_var_set = ComponentSet(matching.values())
    unmatched_vars = []
    for var in igraph.variables:
        if var not in matched_var_set:
            unmatched_vars.append(var)
            print("  %s" % var.name)

    #
    # Plot incidence matrix with matching on the diagonal
    #
    # Need to put the variables in the right order so they appear on the
    # diagonal of the matrix.
    matched_var_list = []
    for con in igraph.constraints:
        if con in matching:
            matched_var_list.append(matching[con])
        else:
            # "Associate" this constraint with a random unmatched var.
            matched_var_list.append(unmatched_vars.pop())
    incidence_matrix = get_structural_incidence_matrix(
        matched_var_list, igraph.constraints
    )
    plt.figure()
    plt.spy(incidence_matrix)
    plt.title("Matching on diagonal, initial")
    if save:
        plt.savefig("matching1.png", transparent=transparent)
    if show:
        plt.show()

    #
    # If we could somehow put flow_mass in sum_component_eqn,
    # we would be done.
    #

    #
    # Use Dulmage-Mendelsohn to determine why system is structurally singular
    #
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    print(len(var_dmp.unmatched))
    print(len(con_dmp.underconstrained), len(var_dmp.underconstrained))
    print(len(con_dmp.square), len(var_dmp.square))
    print(len(con_dmp.overconstrained), len(var_dmp.overconstrained))
    print(len(con_dmp.unmatched))

    #
    # Examine variables and constraints in relevant sets of the partition
    #
    print('unmatched variables:')
    for var in var_dmp.unmatched:
        print("  %s" % var.name)
    print('underconstrained variables:')
    for var in var_dmp.underconstrained:
        print("  %s" % var.name)
    print('underconstraining constraints:')
    for con in con_dmp.underconstrained:
        print("  %s" % con.name)

    print('unmatched constraints:')
    for con in con_dmp.unmatched:
        print("  %s" % con.name)
    print('overconstrained variables:')
    for var in var_dmp.overconstrained:
        print("  %s" % var.name)
    print('overconstraining constraints:')
    for con in con_dmp.overconstrained:
        print("  %s" % con.name)

    #
    # Plot incidence matrix in Dulmage-Mendelsohn order
    #
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
    incidence_matrix = get_structural_incidence_matrix(variables, constraints)
    plt.figure()
    plt.spy(incidence_matrix)
    plt.title("Dulmage-Mendelsohn ordering")
    if save:
        plt.savefig("dmp.png", transparent=transparent)
    if show:
        plt.show()

    #
    # If we replaced the sum mass fraction equation with a sum flow rate
    # equation, it seems like it would solve our problem.
    #
    m.sum_component_eqn.deactivate()

    @m.Constraint()
    def sum_flow_eqn(m):
        return sum(m.flow_mass_comp[j] for j in m.component_list) == m.flow_mass

    #
    # Re-construct IncidenceGraphInterface with new model
    #
    igraph = IncidenceGraphInterface(m)

    #
    # Re-check the model for structural singularity
    #
    matching = igraph.maximum_matching()
    M = len(igraph.constraints)
    N = len(igraph.variables)
    print(M, N, len(matching))
    print("Maximum matching:")
    for con, var in matching.items():
        print()
        print(con.name)
        print(var.name)

    #
    # Plot incidence matrix with zero-free diagonal
    #
    matched_var_list = [matching[con] for con in igraph.constraints]
    incidence_matrix = get_structural_incidence_matrix(
        matched_var_list, igraph.constraints
    )
    plt.figure()
    plt.spy(incidence_matrix)
    plt.title("Matching on diagonal, after fix")
    if save:
        plt.savefig("matching2.png", transparent=transparent)
    if show:
        plt.show()

    #
    # Check numeric singularity
    #
    m._obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    jacobian = nlp.evaluate_jacobian()
    print("Condition number = %1.2e" % np.linalg.cond(jacobian.toarray()))

    #
    # Use block triangularization to try to determine where numeric
    # singularity is coming from
    #
    var_blocks, con_blocks = igraph.get_diagonal_blocks()
    for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
        dim = len(vars)
        print("Block %s, dim = %s" % (i, dim))
        submatrix = nlp.extract_submatrix_jacobian(vars, cons)
        cond = np.linalg.cond(submatrix.toarray())
        print("Condition number = %1.2e" % cond)
        print("  Variables:")
        for var in vars:
            print("    %s" % var.name)
        print("  Constraints:")
        for con in cons:
            print("    %s" % con.name)

    #
    # Plot incidence matrix in block triangular form
    #
    variables = sum(var_blocks, [])
    constraints = sum(con_blocks, [])
    incidence_matrix = get_structural_incidence_matrix(variables, constraints)
    plt.figure()
    plt.spy(incidence_matrix)
    plt.title("Block triangular permutation")
    if save:
        plt.savefig("block_triangular1.png", transparent=transparent)
    if show:
        plt.show()

    #
    # Now we go back to the drawing board
    #
    m.sum_flow_eqn.deactivate()
    m.sum_component_eqn.activate()
    m.particle_porosity.unfix()

    # Need an equation to relate density and flow rate
    @m.Constraint()
    def flow_density_eqn(b):
        return m.flow_mass == m.velocity * m.area * m.dens_mass_particle

    #
    # Make sure we still have zero degrees of freedom and a perfect
    # matching
    #
    print(degrees_of_freedom(m))
    igraph = IncidenceGraphInterface(m)
    matching = igraph.maximum_matching()
    N = len(igraph.constraints)
    M = len(igraph.variables)
    print(N, M, len(matching))

    #
    # Display incidence matrix with matching on the diagonal
    #
    matched_vars = [matching[con] for con in igraph.constraints]
    matrix = get_structural_incidence_matrix(matched_vars, igraph.constraints)
    plt.figure()
    plt.spy(matrix)
    plt.title("Matching on diagonal, after fix")
    if save:
        plt.savefig("matching3.png", transparent=transparent)
    if show:
        plt.show()

    #
    # Check numeric singularity
    #
    nlp = PyomoNLP(m)
    jacobian = nlp.evaluate_jacobian()
    cond = np.linalg.cond(jacobian.toarray())
    print("Condition number = %1.2e" % cond)

    #
    # Use block triangularization to try to determine where numeric
    # singularity is coming from
    #
    var_blocks, con_blocks = igraph.get_diagonal_blocks()
    for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
        dim = len(vars)
        print("Block %s, dim = %s" % (i, dim))
        submatrix = nlp.extract_submatrix_jacobian(vars, cons)
        cond = np.linalg.cond(submatrix.toarray())
        print("Condition number = %1.2e" % cond)
        print("  Variables:")
        for var in vars:
            print("    %s" % var.name)
        print("  Constraints:")
        for con in cons:
            print("    %s" % con.name)

    #
    # Plot incidence matrix in block triangular form
    #
    variables = sum(var_blocks, [])
    constraints = sum(con_blocks, [])
    incidence_matrix = get_structural_incidence_matrix(variables, constraints)
    plt.figure()
    plt.spy(incidence_matrix)
    plt.title("Block triangular permutation, after fix")
    if save:
        plt.savefig("block_triangular2.png", transparent=transparent)
    if show:
        plt.show()

    return igraph


if __name__ == "__main__":
    main()
