import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_components_along_sets
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.subsystems import create_subsystem_block

from pyomo.contrib.incidence_analysis import (
    IncidenceGraphInterface,
    solve_strongly_connected_components,
)

import idaes.core as idaes
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.gas_solid_contactors.unit_models.moving_bed import MBR as MovingBed
from idaes.gas_solid_contactors.properties.methane_iron_OC_reduction import (
    GasPhaseParameterBlock,
    SolidPhaseParameterBlock,
    HeteroReactionParameterBlock,
)

def main():
    m = pyo.ConcreteModel()
    fs_config = {"dynamic": False}
    m.fs = idaes.FlowsheetBlock(default=fs_config)
    m.fs.gas_properties = GasPhaseParameterBlock()
    m.fs.solid_properties = SolidPhaseParameterBlock()
    rxn_config = {
        "solid_property_package": m.fs.solid_properties,
        "gas_property_package": m.fs.gas_properties,
    }
    m.fs.hetero_reactions = HeteroReactionParameterBlock(default=rxn_config)

    nxfe = 10
    xfe_list = [1.0*i/nxfe for i in range(nxfe + 1)]
    mb_config = { 
        "has_holdup": True,
        "finite_elements": nxfe,
        "length_domain_set": xfe_list,
        "transformation_method": "dae.collocation",
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

    m.fs.moving_bed.bed_diameter.fix(6.5)
    m.fs.moving_bed.bed_height.fix(5.0)

    t0 = m.fs.time.first()
    gas_length = m.fs.moving_bed.gas_phase.length_domain
    solid_length = m.fs.moving_bed.solid_phase.length_domain
    x0 = gas_length.first()
    xf = gas_length.last()

    gas = m.fs.moving_bed.gas_phase.properties
    solid = m.fs.moving_bed.solid_phase.properties
    gas[:, :].flow_mol.fix(128.2)
    gas[:, :].pressure.fix(2.0)
    gas[:, :].temperature.fix(298.15)
    gas[:, :].mole_frac_comp["CH4"].fix(0.975)
    gas[:, :].mole_frac_comp["CO2"].fix(0.0)
    gas[:, :].mole_frac_comp["H2O"].fix(0.025)
    solid[:, :].flow_mass.fix(591.4)
    solid[:, :].temperature.fix(1183.15)
    solid[:, :].mass_frac_comp["Fe2O3"].fix(0.45)
    solid[:, :].mass_frac_comp["Fe3O4"].fix(0.0)
    solid[:, :].mass_frac_comp["Al2O3"].fix(0.55)

    solid[t0, xf].particle_porosity.fix(0.27)

    gas_sum_slice = gas[:, :].sum_component_eqn
    solid_sum_slice = solid[:, :].sum_component_eqn
    gas_sum_slice.attribute_errors_generate_exceptions = False
    solid_sum_slice.attribute_errors_generate_exceptions = False
    gas_sum_slice.deactivate()
    solid_sum_slice.deactivate()

    gas = m.fs.moving_bed.gas_phase
    solid = m.fs.moving_bed.solid_phase
    for phase in (gas, solid):
        phase.material_flow_dx_disc_eq.deactivate()
        phase.enthalpy_flow_dx_disc_eq.deactivate()
    gas.pressure_dx_disc_eq.deactivate()

    igraph = IncidenceGraphInterface(m)
    N = len(igraph.constraints)
    M = len(igraph.variables)
    matching = igraph.maximum_matching()
    print(degrees_of_freedom(m))
    print(N, M, len(matching))

    ipopt = pyo.SolverFactory("ipopt")
    ipopt.solve(m, tee=True)

    solve_strongly_connected_components(m)

    ipopt.solve(m, tee=True)


if __name__ == "__main__":
    main()
