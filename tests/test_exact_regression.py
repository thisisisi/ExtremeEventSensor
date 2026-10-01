# Author: Isabella Thiel <thieli@mit.edu>
"""Exact numerical outputs captured in the supplied CPU environment before cleanup."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from numerical_cases import reference_outputs

FIXTURES = Path(__file__).parent / "fixtures"


class ExactRegressionTests(unittest.TestCase):
    def test_fixture_checksums(self):
        for name, expected in json.loads((FIXTURES / "sha256.json").read_text()).items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), expected)

    def test_seeded_models_contexts_and_optimizer_steps(self):
        actual = reference_outputs()
        with np.load(FIXTURES / "cleanup_model_outputs.npz") as expected:
            self.assertEqual(set(actual), set(expected.files))
            for name, value in actual.items():
                with self.subTest(name=name):
                    self.assertEqual(value.dtype, expected[name].dtype)
                    np.testing.assert_array_equal(value, expected[name])


if __name__ == "__main__":
    unittest.main()
