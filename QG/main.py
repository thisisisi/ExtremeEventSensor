#!/usr/bin/env python3
# Author: Isabella Thiel <thieli@mit.edu>
"""QG pipeline: training, checkpoint evaluation, reports, and repeated sweeps."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LEARNING = ROOT / 'learning'


def execute(config, stage, backend='local', dry_run=False):
    """Run stages in independent processes, as in the original batch workflow."""
    stages = ('train', 'evaluate') if stage == 'run' else (stage,)
    if dry_run:
        print(f'{backend}: {" -> ".join(stages)} | {config}')
        return
    if backend == 'local':
        for item in stages:
            subprocess.run([sys.executable, str(LEARNING / f'{item}_qg.py'),
                            '--config_file', str(config)], cwd=LEARNING, check=True)
        return
    
    import sweep
    import yaml
    from datetime import datetime
    
    
    cfg = yaml.safe_load(Path(config).read_text())
    ensemble = cfg.get('ensemble', {})
    meta = dict(submitted_at=datetime.now().isoformat(timespec='seconds'),
                mode='ensemble' if ensemble.get('enabled') else 'deterministic', # switch to deterministic/baseline if ensemble is disabled
                repeat_index=cfg.get('training', {}).get('repeat_index', ''), # give the repeat index for the current run
                seed=cfg.get('training', {}).get('seed', ''), sig=ensemble.get('sig', ''), # specify the ensemble perturbation strength sigma
                n_ens=ensemble.get('n_ens', 0), # specify the number of ensemble members
                Ntime=cfg['data'].get('Ntime_units', ''),  # specify the amount of training time
                strategy=ensemble.get('strategy', 'deterministic')) # specify the ensemble strategy (see also ablations in the paper)
    
    
    os.environ.setdefault("PYTHON_BIN", sys.executable)
    log = sweep.init_sweep_log()
    dependency = None
    for item in stages:
        dependency = sweep.submit_job(config, meta, log, stage=item, dependency=dependency)
        print(f'{item}: submitted job {dependency}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('run', 'train', 'evaluate', 'sweep'):
        sub = actions.add_parser(name, help={
            'run': 'Train, then evaluate the saved checkpoint.',
            'train': 'Train a model.', 'evaluate': 'Evaluate a saved model.',
            'sweep': 'Run the shared experiment grid; preview with --dry-run.'}[name])
        sub.add_argument('--config', type=Path, default=LEARNING / 'ml_config.yaml',
                         help='YAML file; paths inside it are relative to QG/learning.')
        sub.add_argument('--backend', choices=('local', 'slurm'), default='local')
        sub.add_argument('--dry-run', action='store_true', help='Print the plan without running or submitting.')
        if name == 'sweep':
            sub.add_argument('stage', choices=('run', 'train', 'evaluate'), nargs='?', default='run')
            sub.add_argument('--repeats', type=int, default=3)
            sub.add_argument('--repeat-offset', type=int, default=1)
            sub.add_argument('--interval', type=float, default=0, help='Seconds between configurations.')
    
    report = actions.add_parser('report', help='Compare completed runs and generate tables/figures.')
    report.add_argument('--skip-fixed-report', action='store_true')
    report.add_argument('--results-dir', type=Path, help='Saved QG results; relative to your shell directory.')
    report.add_argument('--output-dir', type=Path, help='Report destination; relative to your shell directory.')
    
    args = parser.parse_args()
    if args.action == 'report':
        command = [sys.executable, 'generate_all_comparison_tables.py']
        if args.skip_fixed_report:
            command.append('--skip-fixed-report')
        for name in ('results_dir', 'output_dir'):
            value = getattr(args, name)
            if value is not None:
                command.extend(['--' + name.replace('_', '-'), str(value.resolve())])
        subprocess.run(command, cwd=ROOT / 'evaluation', check=True)
        return
    
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f'Configuration does not exist: {config}')
    sys.path.insert(0, str(LEARNING))
    os.chdir(LEARNING)
    if args.action != 'sweep':
        execute(config, args.action, args.backend, args.dry_run)
        return
    if args.repeats < 1 or args.interval < 0:
        parser.error('--repeats must be positive and --interval nonnegative')
    import sweep
    from sweep_artifacts import model_artifact_path, result_artifact_path
    configs = sweep.generate_sweep_configs(sweep.load_base_config(config),
                                           repeats=args.repeats, repeat_offset=args.repeat_offset)
    print(f'{len(configs)} configurations; grid: QG/learning/sweep.py')
    for index, cfg in enumerate(configs):
        stage = args.stage
        model_exists = Path(model_artifact_path(cfg)).is_file()
        result_exists = Path(result_artifact_path(cfg)).is_file()
        
        # Switch between stages if the requested stage is not possible, and skip if the output already exists.
        if (stage == 'train' and model_exists) or (stage != 'train' and result_exists):
            print(f'{index + 1}: skip existing {stage} output')
            continue
        if stage == 'evaluate' and not model_exists and not args.dry_run:
            raise FileNotFoundError(model_artifact_path(cfg))
        if stage == 'run' and model_exists:
            stage = 'evaluate'
        if args.dry_run:
            print(f'{index + 1}: {args.backend} {stage} | {cfg["ensemble"]} | '
                  f'T={cfg["data"]["Ntime_units"]} repeat={cfg["training"]["repeat_index"]} '
                  f'seed={cfg["training"]["seed"]} train={cfg["data"]["case_name_train"]}')
            continue
        path = sweep.write_config(cfg, f'{time.time_ns()}_{index}', cfg['training']['repeat_index']).resolve()
        execute(path, stage, args.backend)
        if args.interval and index < len(configs) - 1:
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
