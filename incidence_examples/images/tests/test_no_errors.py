import pyomo.common.unittest as unittest

from incidence_examples.images.generate_matching_images import (
    generate_preliminary_images,
    generate_unmatched_variable_images,
    generate_unmatched_constraint_images,
)

class TestNoErrors(unittest.TestCase):
    """
    Some very weak tests to make sure we can run the code without error.
    """

    def test_matching_images(self):
        generate_preliminary_images(save=False, show=False)
        generate_unmatched_variable_images(save=False, show=False)
        generate_unmatched_constraint_images(save=False, show=False)


if __name__ == "__main__":
    unittest.main()
