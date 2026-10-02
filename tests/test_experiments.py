"""Dispatch and entry-point checks for the supported experiment set."""
import copy
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'QG/learning'
sys.path.insert(0, str(CODE))
import experiments
import sweep


class ExperimentTests(unittest.TestCase):
    def test_supported_modules_and_removed_strategies(self):
        expected = {'baseline', 'pooled', 'film', 'random_context', 'coarse_statistics', 'ensemble_mean', 'ensemble_variance', 'covariance_concat', 'mean_spread_concat'}
        self.assertEqual(set(experiments.EXPERIMENTS), expected)
        for strategy, definition in experiments.EXPERIMENTS.items():
            module = definition.module
            self.assertTrue((CODE / f'{module}.py').is_file())
            with patch.object(experiments, 'import_module', return_value=module) as loader:
                self.assertEqual(experiments.load_experiment(strategy), module)
                loader.assert_called_once_with(module)
        for strategy in ('S2', 'S3', 'S5', 'S6', 'S7', 'S8', 'S9', 'unknown'):
            with self.assertRaisesRegex(ValueError, 'Unsupported experiment'):
                experiments.load_experiment(strategy)
        self.assertEqual(list(CODE.glob('ml_funs_qg_s*.py')), [])

    def test_implementation_protocol(self):
        import tensorflow as tf
        for name in experiments.EXPERIMENTS:
            with self.subTest(experiment=name):
                # Each experiment normally runs in its own process.
                with patch.dict(tf.keras.utils.get_custom_objects(), clear=True):
                    implementation = experiments.load_experiment(name)
                self.assertTrue(callable(implementation.train))
                self.assertTrue(callable(implementation.evaluate))

    def test_sweeps_select_supported_experiments(self):
        config = {'data': {}, 'training': {}, 'paths': {}, 'ensemble': {}}
        for grid in (sweep,):
            generated = grid.generate_sweep_configs(copy.deepcopy(config), run_det=False, repeats=1)
            self.assertEqual({cfg['ensemble']['strategy'] for cfg in generated},
                             set(experiments.EXPERIMENTS))
            self.assertEqual(
                {cfg['data']['Ntime_units'] for cfg in generated},
                set(sweep.NTIME_UNITS),
            )
            self.assertEqual(
                {cfg['ensemble']['n_ens'] for cfg in generated},
                set(sweep.N_ENS_LIST),
            )
            self.assertEqual(
                {cfg['ensemble']['sig'] for cfg in generated},
                set(sweep.SIG_LIST),
            )
            expected_count = (
                len(sweep.NTIME_UNITS)
                * len(sweep.SIG_LIST)
                * len(sweep.STRATEGIES)
                * len(sweep.N_ENS_LIST)
            )
            self.assertEqual(len(generated), expected_count)

    def test_names_preserve_checkpoint_and_result_paths(self):
        import artifact_paths
        import sweep_artifacts
        config = sweep.generate_sweep_configs(
            {'data': {}, 'training': {}, 'paths': {}, 'ensemble': {}},
            run_det=False, repeats=1,
        )[0]
        args = ('out', 'case', 'STORN', 'rnudge16.0', 2000, 0.1, 50, 1)
        for name, definition in experiments.EXPERIMENTS.items():
            for function in (artifact_paths.model_save_path2, artifact_paths.results_save_path2):
                self.assertEqual(
                    function(*args, strategy=name, n_members=2, repeat_index=1),
                    function(*args, strategy=definition.artifact_id, n_members=2, repeat_index=1),
                )
            named = copy.deepcopy(config)
            named['ensemble']['strategy'] = name
            saved = copy.deepcopy(named)
            saved['ensemble']['strategy'] = definition.artifact_id
            self.assertEqual(sweep_artifacts.model_artifact_path(named), sweep_artifacts.model_artifact_path(saved))
            self.assertEqual(sweep_artifacts.result_artifact_path(named), sweep_artifacts.result_artifact_path(saved))

    def test_launchers_forward_configuration(self):
        config = {
            'data': {'case_name_train': 'train_case', 'case_name_test': 'test_case',
                     'Ntime_units': 50, 'dt_ml': 0.1},
            'training': {'model_type': 'STORN', 'Nepochs': 2, 'batch_size': 8,
                         'learning_rate': 0.0004},
            'ensemble': {'enabled': True, 'strategy': 'S4', 'n_ens': 2, 'sig': 0.01},
            'paths': {'data_dir': 'input', 'neural_nets_dir': 'weights', 'ml_results_dir': 'results'},
        }
        for enabled in (True, False):
            config['ensemble']['enabled'] = enabled
            for filename, method in [('train_qg.py', 'train'), ('evaluate_qg.py', 'evaluate')]:
                callback = Mock()
                implementation = SimpleNamespace(**{method: callback})
                with patch('utils.load_config', return_value=copy.deepcopy(config)), \
                     patch.object(experiments, 'load_experiment', return_value=implementation) as loader:
                    with self.assertRaises(SystemExit) as stopped:
                        runpy.run_path(str(CODE / filename), run_name='__main__')
                self.assertEqual(stopped.exception.code, 0)
                loader.assert_called_once_with('S4' if enabled else None)
                callback.assert_called_once()
                kwargs = callback.call_args.kwargs
                self.assertEqual(kwargs['T'], 50)
                self.assertEqual(kwargs['data_dir'], 'input')
                self.assertEqual(kwargs['case_name'], 'train_case' if method == 'train' else 'test_case')
                if method == 'train':
                    self.assertEqual(kwargs['sub_sample'], 1)
                    self.assertEqual(kwargs['lr'], 0.0004)
                    self.assertEqual(kwargs['ensemble_cfg'], config['ensemble'] if enabled else None)
                else:
                    self.assertEqual(kwargs['case_name_train'], 'train_case')
                    self.assertEqual(kwargs['ensemble_cfg'], config['ensemble'])


if __name__ == '__main__':
    unittest.main()
