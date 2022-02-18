import pyomo.environ as pyo
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    get_structural_incidence_matrix,
)
from pyomo.dae.flatten import flatten_dae_components
from pyomo.util.subsystems import create_subsystem_block
from pyomo.common.collections import ComponentSet

import idaes.core as idaes
from idaes.core.util.model_statistics import degrees_of_freedom

import matplotlib.pyplot as plt

from incidence_examples.properties import SingularSolidProperties

def main():
    m = pyo.ConcreteModel()
    fs_config = {"dynamic": True, "time_units": pyo.units.s}
    m.fs = idaes.FlowsheetBlock(default=fs_config)
    time = m.fs.time
    t0 = time.first()
    m.fs.properties = SingularSolidProperties()
    cv_config = {"property_package": m.fs.properties}
    m.fs.cv = idaes.ControlVolume0DBlock(default=cv_config)
    m.fs.cv.add_state_blocks(has_phase_equilibrium=False)
    m.fs.cv.add_geometry()
    m.fs.cv.add_material_balances()
    m.fs.cv.add_energy_balances()
    comp_list = m.fs.properties.component_list
    m.fs.cv.flow_mass_comp = pyo.Var(time, comp_list, initialize=1.0)

    @m.fs.cv.Constraint(time, comp_list)
    def flow_comp_eqn(cv, t, j):
        return (
            cv.flow_mass_comp[t, j]
            == cv.properties_out[t0].get_material_flow_terms(None, j)
        )
    disc = pyo.TransformationFactory("dae.finite_difference")
    disc.apply_to(m, wrt=m.fs.time, nfe=1, scheme="BACKWARD")

    # Fix degrees of freedom
    m.fs.cv.volume.fix(1.0)
    for t in time:
        for var in m.fs.cv.properties_in[t].define_state_vars().values():
            var.fix()
    m.fs.cv.properties_out[:].particle_porosity.fix()

    # Fix initial conditions
    m.fs.cv.material_holdup[t0, ...].fix(1.0)
    m.fs.cv.energy_holdup[t0, ...].fix(1.0)

    #
    # Check that we have zero degrees of freedom
    #
    print(degrees_of_freedom(m))

    #
    # Check model for structural singularity
    #
    igraph = IncidenceGraphInterface(m)
    N = len(igraph.constraints)
    M = len(igraph.variables)
    matching = igraph.maximum_matching()
    print(N, M, len(matching))

    #
    # Check subsystems at each point in time for structural singularity
    #
    scalar_vars, dae_vars = flatten_dae_components(m, time, pyo.Var)
    scalar_cons, dae_cons = flatten_dae_components(m, time, pyo.Constraint)

    singular_subsystems = []
    for t in time:
        constraints = [con[t] for con in dae_cons if t in con and con[t].active]
        variables = [var[t] for var in dae_vars if not var[t].fixed]
        n_con = len(constraints)
        n_var = len(variables)
        matching = igraph.maximum_matching(variables, constraints)
        n_matched = len(matching)

        if n_matched != n_var:
            print("Subsystem at %s is structurally singular" % t)
            singular_subsystems.append((constraints, variables))

    constraints, variables = singular_subsystems[0]
    blk = create_subsystem_block(constraints, variables)

    ### This is about where I would like to start.

    #
    # Check system for structural nonsingularity
    #
    igraph = IncidenceGraphInterface(blk)
    matching = igraph.maximum_matching()
    M = len(igraph.constraints)
    N = len(igraph.variables)
    print(M, N, len(matching))
    print("Subsystem at t = %s:" % t0)
    for con, var in matching.items():
        print()
        print(con.name)
        print(var.name)
    print()

    print("Unmatched constraints:")
    for con in constraints:
        if con not in matching:
            print("  %s" % con.name)

    print("Unmatched variables:")
    matched_var_set = ComponentSet(matching.values())
    unmatched_vars = []
    for var in variables:
        if var not in matched_var_set:
            unmatched_vars.append(var)
            print("  %s" % var.name)

    #
    # Plot incidence matrix with matching on the diagonal
    #
    # Need to put the variables in the right order so they appear on the
    # diagonal of the matrix.
    matched_var_list = []
    for con in constraints:
        if con in matching:
            matched_var_list.append(matching[con])
        else:
            # "Associate" this constraint with a random unmatched var.
            matched_var_list.append(unmatched_vars.pop())
    incidence_matrix = get_structural_incidence_matrix(
        matched_var_list, constraints
    )
    plt.spy(incidence_matrix)
    plt.show()

    #
    # If we could somehow put flow_mass in the skeletal density equation,
    # we would be done.
    #

    #
    # Use Dulmage-Mendelsohn to determine why system is structurally singular
    #
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()


if __name__ == "__main__":
    main()
