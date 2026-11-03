# Ensemble-Informed Correction of Coarse Dynamics

This repository uses a small ensemble of intentionally perturbed, low-cost simulations to identify the locally most unstable directions of the dynamics. That information is then used to guide learned corrections to coarse trajectories.

For a light-weight demonstration, see my blog post: [blog](www.mit.edu/~thieli/writing/neurips/).

Full details can be found in the accompanying paper: [arXiv:2609.30746](https://arxiv.org/abs/2609.30746).

The code is organized into two subdirectories, `QG` and `LowDim`, which follow the same overall interface.

## Installation & Getting Started

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) if needed, then create an Python 3.10 environment from the repository root:

```bash
conda create --name extremeeventsensor python=3.10 pip
conda activate extremeeventsensor
bash environment.sh
export MPLBACKEND=Agg
```

`environment.sh` installs the pinned packages from `requirements.txt`. Reuse the environment later with `conda activate extremeeventsensor`.


## Data Availability

LowDim datasets can be generated directly:

```bash
python LowDim/main_create_data.py
```

This creates `LowDim/dataset_tau8.00_sigma1e-06/trajectories.npz` and `metadata.json`, ready for `python main.py lowdim run`. The main simulation control, if one wishes to change the default values, are command-line options:

```bash
python LowDim/main_create_data.py --duration 510 --time-steps 170000 \
  --tau 8 --sigma 1e-6 --runs 2 --members 16 --seed 0
```
where `--duration` is the total time, `--time-steps` is the number of discrete steps, `--tau` and `--sigma` are the system parameters, and `--runs` and `--members` control the number of training/evaluation runs and ensemble members. The random seed can be set with `--seed` for reproducibility.

Use `python LowDim/main_create_data.py --help` for every option. `--coarse-factor` controls temporal coarsening, and `--output PATH` selects a custom destination.

The QG data takes long to generate and therefore is available from [Google Drive](https://drive.google.com/drive/folders/1tCoseAmNCF-Q_F1XCDbblX9YoenjT65D?usp=drive_link).


## Testing of the Code

Run the complete test suite from the repository root to verify that the code is working correctly.

```bash
./test.sh
```

## Running the Code

From the repository root, after preparing the required data:

```bash
python main.py qg run --config QG/learning/ml_config.yaml # QG
python main.py lowdim run # LowDim
```

Both commands train and evaluate the corresponding model.

The original `QG/main.py` and `LowDim/main.py` entry points are better if one wishes to use the full functionality of the code. Use `--help` to see available options, or `run --dry-run` to preview an experiment without launching it.

| Task | Command |
|---|---|
| Evaluate saved QG checkpoints | `python main.py qg evaluate` |
| Generate QG reports | `python main.py qg report --results-dir PATH --output-dir PATH` |
| Generate LowDim summaries | `python main.py lowdim summary --results_root PATH --output_dir PATH` |
| Preview QG sweep | `python main.py qg sweep --dry-run` |
| Preview LowDim sweep | `python main.py lowdim sweep --dry-run` |

### Running parameter sweeps

For QG, edit the grid near the top of [`QG/learning/sweep.py`](QG/learning/sweep.py).
[`SIG_LIST`](QG/learning/sweep.py#L13) controls perturbation strengths, [`STRATEGIES`](QG/learning/sweep.py#L14) controls conditioning methods, [`N_ENS_LIST`](QG/learning/sweep.py#L16) controls ensemble sizes, and [`NTIME_UNITS`](QG/learning/sweep.py#L17) controls training-data lengths. `DET_NTIME_UNITS` contains the deterministic comparison lengths. Then run:

```bash
python main.py qg sweep --dry-run
```
The repeat count and starting index default to `DEFAULT_REPEATS` and `DEFAULT_REPEAT_OFFSET` in `QG/learning/sweep.py`. Use `--repeats N` or `--repeat-offset N` to override either value for one invocation. Remove `--dry-run` to run locally, or add `--backend slurm` to submit to Slurm (recommended).

For LowDim, pass ensemble sizes and conditioning choices directly. `false` uses ensemble mean/spread, `true` uses covariance eigenvectors, and ensemble size `0` runs the unconditioned baseline:

```bash
python main.py lowdim sweep --dry-run \
  --ensemble-sizes 0,2,4,8 --eigencov-values false,true --repeats 3
```

Other LowDim settings, such as `dataset_npz`, `train_time`, `ml_stride`, `epochs`, and `learning_rate`, are shared by every job in one sweep. Put those values in a JSON file and add `--config my_sweep.json`. The available field names and defaults are in `LowDim/corrector/config.py`.

When reproducing experiments, use new output directories so that existing results are not overwritten.

For QG, output locations are controlled by the YAML fields:

```text
paths.neural_nets_dir
paths.ml_results_dir
```

For LowDim, use:

```bash
python main.py lowdim run --dataset PATH --output-dir PATH
```

## Data and Configuration

### QG

Each layer requires:

- Reference/Ground truth fields (\(\psi_1\) and \(\psi_2\)), are the upper and lower-layer streamfunctions:
  - `psi1_ref.mat`
  - `psi2_ref.mat`
- Nudged training fields, for example:
  - `psi1_rnudge16.0.mat` (`rnudge` is the **r**eversed **nudged** /spectrally corrected field with \(\tau=16.0\))
- Coarse / low-fidelity fields used during evaluation:
  - `psi1_coarse.mat`
  - `psi2_coarse.mat`

Conditioned training additionally requires ensemble-member files, for example:

```text
psi1_rnudge16.0_ens01_sig0.01_16.0.mat
```

Equivalent files are required for both layers and for every selected ensemble member.

Fields have shape:

```text
(Ny, Nx, time)
```

The main QG configuration is:

[`QG/learning/ml_config.yaml`](QG/learning/ml_config.yaml)

It controls the cases, model type, training parameters, and file paths. The current defaults include `STORN`, 2,000 epochs, and `dt_ml=0.1`; these correspond to the baseline settings. The default model is unconditioned.

Paths defined in the YAML file are resolved relative to `QG/learning`.

Evaluation and comparison choices are defined in:

[`QG/evaluation/settings.py`](QG/evaluation/settings.py)

### LowDim

Place

```text
trajectories.npz
metadata.json
```

inside

```text
LowDim/dataset_tau8.00_sigma1e-06/
```

or provide a dataset explicitly with `--dataset`.

The NPZ file must contain:

```text
time_reference
time_coarse
reference
coarse
nudged
```

For the three-dimensional system used here, trajectories have shape:

```text
(runs, time, 3)
```

Conditioned training additionally requires:

```text
ensemble
```

with shape:

```text
(runs, members, time, 3)
```

Default LowDim settings are defined in:

[`LowDim/corrector/config.py`](LowDim/corrector/config.py)

The defaults use training/evaluation runs `0/1`, two ensemble members, an attention-based architecture, 40 epochs, and `ml_stride=5`.

You can override configuration fields with a JSON file:

```bash
python main.py lowdim run --config FILE.json
```

For example:

```json
{
  "n_ensemble_members_used": 0
}
```

selects the unconditioned model.

JSON paths are resolved relative to `LowDim`, while command-line paths such as `--dataset` and `--output-dir` are resolved relative to the current shell directory.

## Inference and Outputs

### QG

QG uses every

```python
max(1, int(dt_ml / dt))
```

-th saved frame and preserves the existing exclusion of the final frame.

Outputs are written to:

```text
QG/Neural_Nets      # checkpoints
QG/ML_Results       # evaluation statistics
QG/ML_Plots         # reports and figures
```

### LowDim

LowDim subsamples both training and evaluation using `ml_stride` (default: `5`, corresponding to a nominal spacing of `0.015`).

Corrections are applied over overlapping windows, and overlapping predictions are averaged. The corrected trajectory therefore remains on the sampled time grid.

Results are written under:

```text
LowDim/ML_Results
```

or the directory supplied with `--output-dir`.

Outputs include checkpoints, metrics, plots, and:

```text
corrected_test_outputs.npz
```

## General Notes

The provides *.sh / *.sbatch scripts are intended for SLURM-based HPC clusters. They can be adapted to other environments. In particular, they are writen for MIT's Engaging cluster, which uses the `slurm` scheduler and the `module` system. Please adapt the scripts to your own environment as needed.

## Paper

This repository accompanies:

**Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events**

[arXiv:2609.30746](https://arxiv.org/abs/2609.30746)


```bibtex
@misc{thiel2026mechanismawareensembleconditioning,
      title={Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events},
      author={Isabella S. Thiel and Juan Bello-Rivas and Yannis G. Kevrekidis and Themistoklis P. Sapsis},
      year={2026},
      eprint={2609.30746},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.30746},
}
```

If you use this code, please cite the corresponding paper.
