import pyomo.environ as pyo
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

import idaes.core as idaes
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.gas_solid_contactors.unit_models import MBR as MovingBed
from idaes.gas_solid_contactors.properties.methane_iron_OC_reduction import (
    GasPhaseParameterBlock,
    SolidPhaseParameterBlock,
    HeteroReactionParameterBlock,
)


def main():
    m = pyo.ConcreteModel()
    horizon = 1500.0
    fs_config = {
        "dynamic": True,
        "time_units": pyo.units.s,
        "time_set": [0.0, horizon],
    }
    m.fs = idaes.FlowsheetBlock(default=fs_config)
    m.fs.gas_properties = GasPhaseParameterBlock()
    m.fs.solid_properties = SolidPhaseParameterBlock()
    rxn_config = {
        "solid_property_package": m.fs.solid_properties,
        "gas_property_package": m.fs.gas_properties,
    }
    m.fs.hetero_reactions = HeteroReactionParameterBlock(default=rxn_config)

    nxfe = 10
    nxcp = 1
    xfe_set = [1.0*i/nxfe for i in range(nxfe + 1)]
    mb_config = {
        "has_holdup": True,
        "finite_elements": nxfe,
        "length_domain_set": xfe_set,
        "transformation_method": "dae.collocation",
        "collocation_points": nxcp,
        "transformation_scheme": "LAGRANGE-RADAU",
        "pressure_drop_type": "ergun_correlation",
        "gas_phase_config": {
            "property_package": m.fs.gas_properties,
        },
        "solid_phase_config": {
            "property_package": m.fs.solid_properties,
            "reaction_package": m.fs.hetero_reactions,
        },
    }
    m.fs.moving_bed = MovingBed(default=mb_config)

    time = m.fs.time
    t0 = time.first()
    disc = pyo.TransformationFactory("dae.finite_difference")
    ntfe = 10
    disc.apply_to(m, wrt=time, nfe=ntfe, scheme="BACKWARD")

    # Fix geometry variables
    m.fs.moving_bed.bed_diameter.fix()
    m.fs.moving_bed.bed_height.fix()

    print(degrees_of_freedom(m))

    # Fix dynamic inputs (inlet conditions) at every point in time
    m.fs.moving_bed.gas_inlet.flow_mol[:].fix()
    m.fs.moving_bed.gas_inlet.pressure[:].fix()
    m.fs.moving_bed.gas_inlet.temperature[:].fix()
    m.fs.moving_bed.gas_inlet.mole_frac_comp[:, "CO2"].fix()
    m.fs.moving_bed.gas_inlet.mole_frac_comp[:, "H2O"].fix()
    m.fs.moving_bed.gas_inlet.mole_frac_comp[:, "CH4"].fix()
    m.fs.moving_bed.solid_inlet.flow_mass[:].fix()
    m.fs.moving_bed.solid_inlet.temperature[:].fix()
    m.fs.moving_bed.solid_inlet.particle_porosity[:].fix()
    m.fs.moving_bed.solid_inlet.mass_frac_comp[:, "Fe2O3"].fix()
    m.fs.moving_bed.solid_inlet.mass_frac_comp[:, "Fe3O4"].fix()
    m.fs.moving_bed.solid_inlet.mass_frac_comp[:, "Al2O3"].fix()

    # Fix differential variables at the initial time point
    m.fs.moving_bed.gas_phase.material_holdup[t0, ...].fix()
    m.fs.moving_bed.gas_phase.energy_holdup[t0, ...].fix()
    m.fs.moving_bed.solid_phase.material_holdup[t0, ...].fix()
    m.fs.moving_bed.solid_phase.energy_holdup[t0, ...].fix()

    #
    # We have 8 degrees of freedom!?
    #
    print(degrees_of_freedom(m))

    #
    # What does the Dulmage-Mendelsohn partition tell us?
    #
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()

    print("Overconstrained variables:")
    for var in var_dmp.overconstrained:
        print("  %s" % var.name)
    print("Overconstraining equations:")
    for con in con_dmp.overconstrained:
        print("  %s" % con.name)
    for con in con_dmp.unmatched:
        print("  %s" % con.name)

    x0 = m.fs.moving_bed.length_domain.first()
    xf = m.fs.moving_bed.length_domain.last()

    #
    # Unfix initial conditions that overlap with boundary conditions
    #
    m.fs.moving_bed.gas_phase.material_holdup[t0, x0, ...].unfix()
    m.fs.moving_bed.gas_phase.energy_holdup[t0, x0, ...].unfix()
    m.fs.moving_bed.solid_phase.material_holdup[t0, xf, ...].unfix()
    m.fs.moving_bed.solid_phase.energy_holdup[t0, xf, ...].unfix()

    #
    # Now have zero degrees of freedom
    #
    print(degrees_of_freedom(m))

    #
    # And model has a perfect matching
    #
    igraph = IncidenceGraphInterface(m)
    N = len(igraph.constraints)
    M = len(igraph.variables)
    matching = igraph.maximum_matching()
    print(N, M, len(matching))

    #
    # What if we didn't check the Dulmage-Mendelsohn partition and just
    # assumed that initial conditions at x0 needed to be unfixed?
    #
    # Fix solid phase initial conditions at x0
    m.fs.moving_bed.solid_phase.material_holdup[t0, x0, ...].unfix()
    m.fs.moving_bed.solid_phase.energy_holdup[t0, x0, ...].unfix()
    # And unfix them at xf
    m.fs.moving_bed.solid_phase.material_holdup[t0, xf, ...].fix()
    m.fs.moving_bed.solid_phase.energy_holdup[t0, xf, ...].fix()

    #
    # Degrees of freedom look good!
    #
    print(degrees_of_freedom(m))

    #
    # But the Jacobian doesn't have a perfect matching, so it is guaranteed
    # to be singular.
    #
    igraph = IncidenceGraphInterface(m)
    N = len(igraph.constraints)
    M = len(igraph.variables)
    matching = igraph.maximum_matching()
    print(N, M, len(matching))

    #
    # So we really needed to unfix solid initial conditions only at x0
    #
    m.fs.moving_bed.solid_phase.material_holdup[t0, x0, ...].fix()
    m.fs.moving_bed.solid_phase.energy_holdup[t0, x0, ...].fix()
    m.fs.moving_bed.solid_phase.material_holdup[t0, xf, ...].unfix()
    m.fs.moving_bed.solid_phase.energy_holdup[t0, xf, ...].unfix()

    #
    # As we already have seen, this gives us zero degrees of freedom and a
    # perfect matching.
    #
    print(degrees_of_freedom(m))
    igraph = IncidenceGraphInterface(m)
    N = len(igraph.constraints)
    M = len(igraph.variables)
    matching = igraph.maximum_matching()
    print(N, M, len(matching))

    #
    # Take-aways from this example:
    # - Just because you have zero degrees of freedom doesn't mean your
    #   specification is "correct"
    # - Dulmage-Mendelsohn can help when we expect to have zero degrees
    #   of freedom, but don't
    # - Always check whether your (square) model has a perfect matching
    #   of variables and equations
    # - Dulmage-Mendelsohn can help you debug when it doesn't
    #

if __name__ == "__main__":
    main()
