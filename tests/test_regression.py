"""Seeded numerical fixtures captured before the repository reorganization."""
# Author: Isabella Thiel <thieli@mit.edu>
import unittest
from pathlib import Path
import numpy as np
from numerical_cases import reference_outputs


class NumericalRegressionTests(unittest.TestCase):
    def test_reference_outputs(self):
        actual = reference_outputs()
        with np.load(Path(__file__).parent / 'fixtures/reference_outputs.npz') as expected:
            self.assertEqual(set(actual), set(expected.files))
            for name, value in actual.items():
                with self.subTest(name=name):
                    # Allow floating-point library/CPU differences on other machines.
                    np.testing.assert_allclose(value, expected[name], rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
