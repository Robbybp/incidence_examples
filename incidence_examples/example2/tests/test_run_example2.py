import pyomo.common.unittest as unittest
import pyomo.environ as pyo

import incidence_examples.example2.run_scc_example as example_script


class TestRunExample(unittest.TestCase):

    def test_solve_status(self):
        res1, res2 = example_script.main(nxfe=2)
        self.assertIs(
            res1.solver.termination_condition,
            pyo.TerminationCondition.infeasible,
        )
        pyo.assert_optimal_termination(res2)


if __name__ == "__main__":
    unittest.main()
