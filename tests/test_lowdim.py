# Author: Isabella Thiel <thieli@mit.edu>
"""Small synthetic-data checks; no archived data or accelerator required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LowDim'))
from corrector.config import CorrectorConfig
from corrector import data
from corrector.model import ResidualTCNCorrector, ResidualAttentionCorrector
from main import load_config


class LowDimTests(unittest.TestCase):
    def test_dataset_generator_writes_training_layout(self):
        sys.path.insert(0, str(ROOT / "LowDim"))
        from data_generation import DatasetConfig, generate_dataset

        config = DatasetConfig(
            t_span=(0.0, 0.06), n_ref=21, n_runs=2,
            ensemble_size=2, sigma=1e-6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = generate_dataset(config, Path(tmp) / "dataset", seed=7)
            self.assertTrue((output / "metadata.json").is_file())
            with np.load(output / "trajectories.npz") as data:
                self.assertEqual(data["reference"].shape, (2, 5, 3))
                self.assertEqual(data["ensemble"].shape, (2, 2, 5, 3))
                self.assertTrue(np.isfinite(data["ensemble"]).all())

    def test_reference_interpolation_and_explicit_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dataset' / 'trajectories.npz'
            path.parent.mkdir()
            time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            expected = np.column_stack([time, 2 * time + 1, -time])
            reference = expected[[0, 2, 4]][None]
            np.savez(path, time_reference=time[[0, 2, 4]], time_coarse=time,
                     reference=reference, nudged=expected[None], coarse=expected[None])
            cfg = CorrectorConfig(dataset_npz=str(path), n_ensemble_members_used=0,
                                  run_id_test=0)
            _, _, target, context = data.load_train_data(cfg)
            np.testing.assert_array_equal(target, expected.astype(np.float32))
            self.assertIsNone(context)
            self.assertEqual(cfg.output_root(), Path(tmp) / 'ML_Results')
            cfg.results_dir = str(Path(tmp) / 'separate-results')
            self.assertEqual(cfg.output_root(), Path(tmp) / 'separate-results')

    def test_configuration_and_repeat_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / 'dataset_sigma1e-06' / 'trajectories.npz'
            dataset.parent.mkdir()
            np.savez(dataset, time_coarse=np.array([0.0, 0.015, 0.03]))
            cfg = CorrectorConfig(dataset_npz=str(dataset))
            self.assertEqual(cfg.ml_dt, 0.015)
            self.assertEqual(cfg.train_interval_length, 333)
            self.assertEqual(cfg.sigma_value, 1e-6)
            self.assertEqual(cfg.output_stem(), 'sigma1.00e-06_det_attention_ens2_vec1_stride1_T5_run0_1')
            first = cfg.resolve_output_dir()
            first.mkdir(parents=True)
            self.assertTrue(cfg.resolve_output_dir().name.endswith('_rep001'))
            override = Path(tmp) / 'config.json'
            override.write_text(json.dumps({'epochs': 1, 'n_ensemble_members_used': 0}))
            with patch.dict('os.environ', {'LOWDIM_CONFIG_JSON': str(override)}):
                self.assertEqual(load_config().epochs, 1)
                override.write_text('{"unknown_parameter": 1}')
                with self.assertRaises(ValueError):
                    load_config()

    def test_data_windows_and_models(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trajectories.npz'
            trajectory = rng.normal(size=(2, 20, 3)).astype(np.float32)
            ensemble = trajectory[:, None] + rng.normal(size=(2, 3, 20, 3)) * 0.1
            np.savez(path, time_reference=np.arange(20), time_coarse=np.arange(20),
                     reference=trajectory, coarse=trajectory + 0.2,
                     nudged=trajectory + 0.1, ensemble=ensemble)
            for members in (0, 2):
                cfg = CorrectorConfig(dataset_npz=str(path), n_ensemble_members_used=members)
                _, x, target, context = data.load_train_data(cfg)
                np.testing.assert_array_equal(target, trajectory[0])
                self.assertEqual(x.dtype, np.float32)
                _, coarse = data.load_test_coarse_data(cfg)
                _, test_reference = data.load_test_reference_data(cfg)
                np.testing.assert_array_equal(test_reference, trajectory[1])
                np.testing.assert_array_equal(coarse, trajectory[1] + 0.2)
                mean, std = data.compute_data_statistics(target)
                normalized = data.normalize(x, mean, std)
                np.testing.assert_allclose(data.denormalize(normalized, mean, std), x, atol=2e-7)
                windows, contexts = data.make_windows(normalized, context, 8)
                self.assertEqual(windows.shape, (13, 8, 3))
                np.testing.assert_array_equal(windows[1], normalized[1:9])
                for cls in (ResidualTCNCorrector, ResidualAttentionCorrector):
                    tf.keras.utils.set_random_seed(19)
                    kwargs = dict(state_dim=3, hidden_dim=8, context_dim=0 if context is None else 3,
                                  film_hidden_dim=4)
                    if cls is ResidualAttentionCorrector:
                        kwargs.update(num_heads=2, num_layers=1, dropout_rate=0)
                    model = cls(**kwargs)
                    output = model(tf.constant(windows[:2]), None if contexts is None else tf.constant(contexts[:2]))
                    self.assertEqual(output.shape, (2, 8, 3))
                    self.assertTrue(np.isfinite(output.numpy()).all())
                    prediction = data.apply_sliding_window_inference(model, normalized, 8, context)
                    self.assertEqual(prediction.shape, (20, 3))
                    self.assertTrue(np.isfinite(prediction).all())
            data.save_run_metadata(cfg, Path(tmp) / 'result_rep003')
            self.assertIn('resolved_repeat_index: 3', (Path(tmp) / 'result_rep003/config.txt').read_text())

    def test_one_epoch_pipeline_outputs(self):
        from corrector.training import train_and_evaluate
        rng = np.random.default_rng(29)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / 'trajectories.npz'
            reference = rng.normal(size=(2, 48, 3)).astype(np.float32)
            np.savez(dataset, time_reference=np.arange(48) * 0.003,
                     time_coarse=np.arange(48) * 0.003, reference=reference,
                     coarse=reference + 0.2, nudged=reference + 0.1,
                     ensemble=reference[:, None] + rng.normal(size=(2, 2, 48, 3)) * 0.1)
            cfg = CorrectorConfig(dataset_npz=str(dataset), epochs=1, batch_size=16,
                                 train_time=1, test_time=1, ml_stride=1,
                                 train_sequence_length=8, hidden_dim=8, film_hidden_dim=4,
                                 attention_heads=2, attention_layers=1, quantile_grid_size=21)
            out = Path(tmp) / 'result_rep000'
            data.save_run_metadata(cfg, out)
            tf.keras.utils.set_random_seed(31)
            train_and_evaluate(cfg, out)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['test_context_source'], 'coarse_rolling')
            self.assertTrue((out / 'lsde_film_ckpt.index').exists())
            self.assertTrue((out / 'training_history.csv').exists())
            with np.load(out / 'corrected_test_outputs.npz') as result:
                self.assertEqual(set(result.files), {'time', 'reference', 'coarse', 'corrected',
                    'corrected_before_calibration', 'state_source_quantiles', 'state_target_quantiles'})
                self.assertEqual(result['corrected'].shape, (48, 3))
                self.assertTrue(np.isfinite(result['corrected']).all())

    def test_overlap_covers_final_samples(self):
        class Identity:
            def __call__(self, x):
                return x
        x = np.arange(57, dtype=np.float32).reshape(19, 3)
        np.testing.assert_array_equal(data.apply_sliding_window_inference(Identity(), x, 8), x)
        with self.assertRaises(ValueError):
            data.make_windows(x, None, 20)


if __name__ == '__main__':
    unittest.main()
