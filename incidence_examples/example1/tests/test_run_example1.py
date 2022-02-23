import pyomo.common.unittest as unittest

import incidence_examples.example1.run_clc_dm_example as example_script


class TestRunExample(unittest.TestCase):

    def test_dof_and_matching(self):
        igraph = example_script.main(nxfe=2, ntfe=2)
        M = len(igraph.constraints)
        N = len(igraph.variables)
        self.assertEqual(N, M)
        matching = igraph.maximum_matching()
        self.assertEqual(N, len(matching))


if __name__ == "__main__":
    unittest.main()
