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


def execute(config, stage, backend='local', dry_run=False, log_submissions=True):
    """Run stages in independent processes, as in the original batch workflow."""
    stages = ('train', 'evaluate') if stage == 'run' else (stage,)
    if dry_run:
        print(f'{backend}: {" -> ".join(stages)} | {config}')
        return
    if backend == 'local':
        for item in stages:
            subprocess.run([sys.executable, str(LEARNING / f'{item}_qg.py'),
                            '--config_file', str(config)], cwd=LEARNING, check=True)
        return []
    
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
    submitted = []
    for item in stages:
        dependency = sweep.submit_job(config, meta, log, stage=item, dependency=dependency)
        submitted.append((item, dependency))
        if log_submissions:
            print(f'{item}: submitted job {dependency}', flush=True)
    return submitted


def sweep_config_label(cfg):
    """Show the parameters that identify one sweep configuration."""
    ensemble = cfg.get('ensemble', {})
    enabled = ensemble.get('enabled', False)
    n_add = int(ensemble.get('n_ens', 0)) if enabled else 0
    n_members = n_add
    if enabled and ensemble.get('strategy') == 'pooled':
        n_members += int(ensemble.get('include_baseline', True))
    if enabled:
        from experiments import artifact_id
        strategy = f"{ensemble['strategy']} ({artifact_id(ensemble['strategy'])})"
        sig = ensemble.get('sig', '?')
    else:
        strategy = 'deterministic'
        sig = cfg['data']['case_name_train'].rsplit('_sig', 1)[-1]
    return (f"sig={sig} strategy={strategy} N_add={n_add} N_members={n_members} "
            f"T_train={cfg['data']['Ntime_units']} "
            f"repeat={cfg['training']['repeat_index']} seed={cfg['training']['seed']}")


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
            sub.add_argument('--repeats', type=int, help='Overrides DEFAULT_REPEATS in learning/sweep.py.')
            sub.add_argument('--repeat-offset', type=int, help='Overrides DEFAULT_REPEAT_OFFSET in learning/sweep.py.')
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
    import sweep
    repeats = sweep.DEFAULT_REPEATS if args.repeats is None else args.repeats
    repeat_offset = sweep.DEFAULT_REPEAT_OFFSET if args.repeat_offset is None else args.repeat_offset
    if repeats < 1 or args.interval < 0:
        parser.error('--repeats must be positive and --interval nonnegative')
    from sweep_artifacts import model_artifact_path, result_artifact_path
    base_config = sweep.load_base_config(config)
    configs = sweep.generate_sweep_configs(base_config,
                                           repeats=repeats, repeat_offset=repeat_offset)
    print(f'=== QG SWEEP: {len(configs)} configurations | repeats={repeats} '
          f'| repeat_offset={repeat_offset} | stage={args.stage} | backend={args.backend} ===', flush=True)
    print('Grid: QG/learning/sweep.py | Base settings: ' + str(config), flush=True)
    print(f'Grid values: sig={sweep.SIG_LIST} strategies={sweep.STRATEGIES} '
          f'N_add={sweep.N_ENS_LIST} T_train={sweep.NTIME_UNITS} '
          f'deterministic={sweep.RUN_DET} T_det={sweep.DET_NTIME_UNITS} '
          f'base_seed={sweep.BASE_SEED} (seed=base_seed+repeat)', flush=True)
    print('Sweep replaces YAML train/test case names, Ntime_units, ensemble settings, '
          'training.repeat_index, and training.seed.', flush=True)
    print('Existing outputs are matched by repeat index, not seed; changing BASE_SEED '
          'alone does not resubmit completed repeats.', flush=True)
    print(f"Model={configs[0]['training']['model_type']} epochs={configs[0]['training']['Nepochs']} "
          f"T_test={configs[0]['data']['test_time_units']} "
          f"test_case={configs[0]['data']['case_name_test']}", flush=True)
    submitted_count = 0
    for index, cfg in enumerate(configs):
        stage = args.stage
        label = sweep_config_label(cfg)
        position = f'[{index + 1}/{len(configs)}]'
        model_exists = Path(model_artifact_path(cfg)).is_file()
        result_exists = Path(result_artifact_path(cfg)).is_file()
        
        # Switch between stages if the requested stage is not possible, and skip if the output already exists.
        if (stage == 'train' and model_exists) or (stage != 'train' and result_exists):
            print(f'{position} skip existing {stage} | {label}', flush=True)
            continue
        if stage == 'evaluate' and not model_exists and not args.dry_run:
            raise FileNotFoundError(model_artifact_path(cfg))
        if stage == 'run' and model_exists:
            stage = 'evaluate'
        if args.dry_run:
            print(f'{position} preview {stage} | {label}', flush=True)
            continue
        path = sweep.write_config(cfg, f'{time.time_ns()}_{index}', cfg['training']['repeat_index']).resolve()
        submitted = execute(path, stage, args.backend, log_submissions=False)
        if args.backend == 'local':
            print(f'{position} completed {stage} | {label}', flush=True)
        for job_stage, job_id in submitted:
            submitted_count += 1
            print(f'{position} submitted {job_stage} job={job_id} | {label}', flush=True)
            if submitted_count % 25 == 0:
                print(f'=== {submitted_count} Slurm jobs submitted; '
                      f'{index + 1}/{len(configs)} configurations checked ===', flush=True)
        if args.interval and index < len(configs) - 1:
            time.sleep(args.interval)
    if not args.dry_run and args.backend == 'slurm':
        print(f'=== SWEEP SUBMISSION COMPLETE: {submitted_count} Slurm jobs submitted; '
              f'{len(configs)} configurations checked ===', flush=True)


if __name__ == '__main__':
    main()
