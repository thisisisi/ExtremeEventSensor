"""QG configuration, MATLAB I/O, context, and checkpoint contracts."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io as sio
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'QG/learning'))
import utils
# Strategies normally run in separate processes; isolate their custom-layer registry.
with patch.dict(tf.keras.utils.get_custom_objects(), clear=True):
    import baseline
import film
from loss_params_qg import loss_params_2D


class QGTests(unittest.TestCase):
    def test_config_and_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('ensemble:\n  enabled: true\n  strategy: S4\n')
            with patch.object(sys, 'argv', ['train', '--config_file', str(path)]):
                self.assertEqual(utils.load_config()['ensemble']['strategy'], 'S4')
        args = ('out', 'case', 'LSTM', 'rnudge16.0', 2, 0.1, 50, 1)
        self.assertEqual(utils.model_save_path2(*args, strategy='S4', n_members=2, repeat_index=3),
                         'out/case_LSTM_rnudge16.0_epochs_2_dt_0.1_T_50_j1_SS4-k2_rep003.h5')
        self.assertEqual(utils.results_save_path2(*args, strategy='S4', n_members=2, repeat_index=3),
                         'out/case_LSTM_rnudge16.0_loss_L2_epochs_2_dt_0.1_T_50_Nens_1_SS4-k2_rep003_Ensemble_Stats_Big.mat')
        for strategy in baseline.CONTROL_STRATEGIES:
            baseline._validate_control_design(strategy, 50, 0.01, 2)
            with self.assertRaises(ValueError):
                baseline._validate_control_design(strategy, 100, 0.01, 2)

    def test_matlab_loading_and_slicing(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / 'case'
            case.mkdir()
            field = np.arange(80).reshape(2, 2, 20)
            sio.savemat(case / 'psi1_rnudge16.0.mat', {'psi1_rnudge': field})
            sio.savemat(case / 'psi1_rnudge16.0_ens01_sig0.01_16.0.mat', {'member': field})
            sio.savemat(case / 'psi2_rnudge_ens01_16.0.mat', {'member': field})
            np.testing.assert_array_equal(utils.load_field(tmp, 'case', 'rnudge16.0', 1), field)
            np.testing.assert_array_equal(utils.load_member_field(tmp, 'case', 1, 1, basis_type='rnudge16.0', sig='0.01'), field)
            np.testing.assert_array_equal(utils.load_member_field(tmp, 'case', 2, 1, basis_type='rnudge16.0', sig='0.01'), field)
        self.assertEqual(utils.compute_skip_and_length({'data_times': 20, 'tstart': 0}, 0.2, 0.5, 0.1), (2, 6))
        np.testing.assert_array_equal(film._make_window_indices(9, 3, 2, 3), [[0, 2, 4], [3, 5, 7]])

    def test_covariance_context_is_causal(self):
        rng = np.random.default_rng(21)
        coarse = rng.normal(size=(2, 2, 12, 2)).astype(np.float32)
        context = baseline._context_from_coarse_causal(coarse, n_modes=1, window=4)
        coarse[:, :, 7:] += 100
        changed = baseline._context_from_coarse_causal(coarse, n_modes=1, window=4)
        np.testing.assert_array_equal(context[:7], changed[:7])
        np.testing.assert_allclose(np.linalg.norm(context[1:], axis=1), 1, atol=2e-7)
        a = baseline._build_control_test_context('S10', coarse, 8, 1, 4, 12)
        b = baseline._build_control_test_context('S10', coarse, 8, 1, 4, 12)
        np.testing.assert_array_equal(a, b)

    def test_storn_forward(self):
        from loss_params_qg import loss_params_2DV
        tf.keras.utils.set_random_seed(37)
        alpha = loss_params_2DV(1, 1, 2, 2, 2, 0.1, 4)
        model = film.build_film_storn(8, 8, 4, alpha, inter_dim=8, latent_dim=4)
        result = model({'field_in': tf.ones((2, 5, 8)), 'cond_in': tf.ones((2, 5, 4))})['field']
        self.assertEqual(result.shape, (2, 5, 16))  # field + latent mean + latent scale
        self.assertTrue(np.isfinite(result.numpy()).all())

    def test_film_forward_training_and_checkpoint(self):
        tf.keras.utils.set_random_seed(23)
        alpha = loss_params_2D(1, 1, 2, 2, 2, 0.1)
        x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 4, 8)
        z = np.ones((2, 4, 4), dtype=np.float32)
        model = film.build_film_lstm(8, 8, 4, alpha)
        inputs = {'field_in': x, 'cond_in': z}
        self.assertEqual(model(inputs)['field'].shape, (2, 4, 8))
        loss = model.train_on_batch(inputs, {'field': x})
        self.assertTrue(np.isfinite(loss).all())
        expected = model(inputs)['field'].numpy()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'model.h5')
            model.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_array_equal(restored(inputs)['field'].numpy(), expected)


if __name__ == '__main__':
    unittest.main()
