import pyomo.common.unittest as unittest

import incidence_examples.tutorial.run_tutorial as tutorial_script


class TestRunTutorial(unittest.TestCase):

    def test_incidence(self):

        igraph = tutorial_script.main(show=False, save=False)
        M = len(igraph.constraints)
        N = len(igraph.variables)
        matching = igraph.maximum_matching()
        self.assertEqual(M, N)
        self.assertEqual(N, len(matching))


if __name__ == "__main__":
    unittest.main()
