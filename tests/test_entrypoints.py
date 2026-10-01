# Author: Isabella Thiel <thieli@mit.edu>
"""Public workflow commands: process isolation, dependencies, and safe previews."""
import importlib.util
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qg_cli', ROOT / 'QG/main.py')
qg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qg)


class EntrypointTests(unittest.TestCase):
    def test_importing_comparison_does_not_generate_reports(self):
        sys.path.insert(0, str(ROOT / 'QG/evaluation'))
        with patch('metrics.run_all_metrics_for_configs') as calculate, \
             patch('json_2_latex_structured.build_master_table') as table, \
             patch('json_2_latex_structured.build_master_table_relative') as relative:
            runpy.run_path(str(ROOT / 'QG/evaluation/compare_all.py'), run_name='import_check')
        calculate.assert_not_called()
        table.assert_not_called()
        relative.assert_not_called()

    def test_report_paths_resolve_before_worker_changes_directory(self):
        with patch.object(sys, 'argv', ['main.py', 'report', '--results-dir', 'input cases',
                                      '--output-dir', 'new reports']), \
             patch.object(qg.subprocess, 'run') as run:
            qg.main()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--results-dir') + 1], str(Path('input cases').resolve()))
        self.assertEqual(command[command.index('--output-dir') + 1], str(Path('new reports').resolve()))
        self.assertEqual(run.call_args.kwargs['cwd'], qg.ROOT / 'evaluation')

    def test_qg_pipeline_uses_separate_processes_and_stops_on_failure(self):
        with patch.object(qg.subprocess, 'run') as run:
            qg.execute('/tmp/config.yaml', 'run')
            self.assertEqual([Path(call.args[0][1]).name for call in run.call_args_list],
                             ['train_qg.py', 'evaluate_qg.py'])
            self.assertTrue(all(call.kwargs['cwd'] == qg.LEARNING for call in run.call_args_list))
        with patch.object(qg.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'train')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                qg.execute('/tmp/config.yaml', 'run')
            self.assertEqual(run.call_count, 1)

    def test_slurm_evaluation_depends_on_training(self):
        sys.path.insert(0, str(qg.LEARNING))
        import sweep
        with patch.object(sweep, 'init_sweep_log', return_value='/tmp/runtime.csv'), \
             patch.object(sweep, 'submit_job', side_effect=['123', '124']) as submit:
            qg.execute(qg.LEARNING / 'ml_config.yaml', 'run', 'slurm')
        self.assertEqual(submit.call_args_list[0].kwargs, {'stage': 'train', 'dependency': None})
        self.assertEqual(submit.call_args_list[1].kwargs, {'stage': 'evaluate', 'dependency': '123'})

    def test_preview_does_not_launch_workers(self):
        with patch.object(qg.subprocess, 'run') as run:
            qg.execute('/tmp/config.yaml', 'run', 'slurm', dry_run=True)
            run.assert_not_called()

    def test_public_help_and_lowdim_preview_without_writes(self):
        for project in ('QG', 'LowDim'):
            completed = subprocess.run([sys.executable, str(ROOT / project / 'main.py'), '--help'],
                                       capture_output=True, text=True, check=True)
            self.assertIn('sweep', completed.stdout)
        subprocess.run([sys.executable, str(ROOT / 'LowDim/main.py'), 'run',
                        '--backend', 'slurm', '--dry-run'],
                       capture_output=True, text=True, check=True)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / 'LowDim/main.py'), 'sweep',
                            '--dry-run', '--ensemble-sizes', '0,2', '--repeats', '1',
                            '--resubmit_existing', '--config_dir', tmp],
                           capture_output=True, text=True, check=True)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
