# Author: Isabella Thiel <thieli@mit.edu>
"""Local CLI and numerical regression checks with isolated outputs."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repository_cli", ROOT / "main.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class RootCliTests(unittest.TestCase):
    def test_dispatch_preserves_arguments_and_exit_status(self):
        for arguments, script, forwarded in (
            (["qg", "run", "--dry-run"], "QG/main.py", ["run", "--dry-run"]),
            (["lowdim", "run", "--config", "a b.json"], "LowDim/main.py",
             ["run", "--config", "a b.json"]),
        ):
            with self.subTest(arguments=arguments), patch.object(cli, "run_script", return_value=7) as run:
                self.assertEqual(cli.main(arguments), 7)
                run.assert_called_once_with(script, forwarded)

    def test_lowdim_cli_synthetic_one_epoch(self):
        """Synthetic wiring check; it is not evidence of research reproduction."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(29)
            reference = rng.normal(size=(2, 48, 3)).astype(np.float32)
            np.savez(tmp / "synthetic.npz", time_reference=np.arange(48) * 0.003,
                     time_coarse=np.arange(48) * 0.003, reference=reference,
                     coarse=reference + 0.2, nudged=reference + 0.1)
            config = {
                "n_ensemble_members_used": 0, "epochs": 1, "batch_size": 16,
                "train_time": 1, "test_time": 1, "ml_stride": 1,
                "train_sequence_length": 8, "hidden_dim": 8, "film_hidden_dim": 4,
                "attention_heads": 2, "attention_layers": 1, "quantile_grid_size": 21,
                "repeat_index": 0,
            }
            (tmp / "config.json").write_text(json.dumps(config))
            subprocess.run(
                [sys.executable, str(ROOT / "main.py"), "lowdim", "run",
                 "--config", "config.json", "--dataset", "synthetic.npz",
                 "--output-dir", "outputs"], cwd=tmp, check=True,
                env=dict(os.environ, MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES=""),
                capture_output=True, text=True,
            )
            runs = list((tmp / "outputs").iterdir())
            self.assertEqual(len(runs), 1)
            with np.load(runs[0] / "corrected_test_outputs.npz") as result:
                np.testing.assert_array_equal(result["reference"], reference[1])
                self.assertEqual(result["corrected"].shape, (48, 3))
                self.assertTrue(np.isfinite(result["corrected"]).all())
            self.assertTrue((runs[0] / "summary.json").is_file())
            self.assertFalse((tmp / "ML_Results").exists())


if __name__ == "__main__":
    unittest.main()
