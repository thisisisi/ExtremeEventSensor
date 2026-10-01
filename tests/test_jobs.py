"""Exercise job setup without submitting jobs or training a model."""
import csv
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'QG/learning'


class JobTests(unittest.TestCase):
    def test_lowdim_sweep_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'python'
            arguments = Path(tmp) / 'arguments.txt'
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_ARGUMENTS"\n')
            executable.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(executable), TEST_ARGUMENTS=str(arguments))
            subprocess.run(['bash', str(ROOT / 'LowDim/cluster/sweep_submit.sh'), '--dry-run', '0'],
                           cwd=tmp, env=env, check=True)
            args = arguments.read_text().splitlines()
            self.assertIn('--dry_run', args)
            self.assertEqual(args[args.index('--repeats') + 1], '1')
            self.assertEqual(args[args.index('--submit_interval_seconds') + 1], '0')

    def test_configuration_snapshot_and_runtime_log(self):
        for job in ('train.sh', 'evaluate.sh'):
            with self.subTest(job=job), tempfile.TemporaryDirectory() as tmp:
                cwd = Path(tmp)
                for name in (job, 'job_runtime.sh'):
                    (cwd / 'cluster').mkdir(exist_ok=True)
                    shutil.copy2(ROOT / 'QG/cluster' / name, cwd / 'cluster' / name)
                work = cwd / 'learning'
                work.mkdir()
                config = 'ensemble:\n  enabled: true\n  strategy: film\n'
                (work / 'ml_config.yaml').write_text(config)
                env = dict(os.environ, PYTHON_BIN=shutil.which('true'), SLURM_JOB_ID='test',
                           PROJECT_DIR=str(work), SWEEP_LOG_CSV=str(cwd / 'runtime.csv'), SWEEP_STRATEGY='film')
                # Stub site-specific commands; the Python executable is a no-op.
                command = 'module() { :; }; sacct() { :; }; export -f module sacct; bash "$1"'
                subprocess.run(['bash', '-c', command, 'job-test', str(cwd / 'cluster' / job)], cwd=cwd, env=env,
                               capture_output=True, text=True, check=True)
                self.assertEqual((work / 'runs/ml_config.test.yaml').read_text(), config)
                with (cwd / 'runtime.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['strategy'], 'film')
                self.assertEqual(rows[0]['slurm_job_id'], 'test')
                self.assertGreaterEqual(int(rows[0]['elapsed_seconds']), 0)


if __name__ == '__main__':
    unittest.main()
