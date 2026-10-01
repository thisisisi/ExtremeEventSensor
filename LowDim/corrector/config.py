# Author: Isabella Thiel <thieli@mit.edu>
"""Scientific defaults and output naming for the LowDim corrector."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorrectorConfig:
    # Run mode:
    # - Non-ensemble baseline: n_ensemble_members_used = 0
    # - FiLM ensemble-informed training: n_ensemble_members_used > 0
    # Test-time correction always uses the coarse trajectory only.

    # Data selection
    dataset_npz: str = "dataset_tau8.00_sigma1e-06/trajectories.npz"
    results_dir: str | None = None  # None preserves the archive-relative output root.
    run_id_train: int = 0
    run_id_test: int = 1
    n_ensemble_members_used: int = 2
    ml_stride: int = 5

    # Repeated runs with identical hyperparameters.
    # If repeat_index is None, a new training run uses the next free _repXXX folder.
    repeat_index: int | None = None
    auto_increment_repeats: bool = True

    # Model selection
    model_kind: str = "deterministic_residual"

    # Conditioning strategy
    eigencov: bool = True  # Use eigenvectors of ensemble covariance instead of mean/std
    n_eigcov_modes: int = 1  # Number of top eigenvectors to use if eigencov=True

    # Time slicing controls, time step is 0.003
    train_time: int = 5
    test_time: int = 500
    train_sequence_length: int = 128

    # Model architecture
    state_dim: int = 3
    model_architecture: str = "attention"  # "attention" or "tcn"
    hidden_dim: int = 96
    film_hidden_dim: int = 48
    attention_heads: int = 4
    attention_layers: int = 3
    ffn_multiplier: int = 2
    dropout_rate: float = 0.05

    # Optimization
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 8e-4

    # Tail weighting and checkpoint selection.
    tail_loss_weight: float = 10.0      # Validation tail-MSE weight
    tail_threshold_std: float = 2.0    # Threshold in units of std of reference z3
    tail_weight_factor: float = 3.0    # Extra weight during training on tail timesteps

    # Hybrid loss weights for both residual architectures
    lambda_delta: float = 1e-3          # Regularization on residual ||delta||^2
    lambda_moment: float = 5e-2         # Weight on mean/std matching
    lambda_tail: float = 5e-2           # Weight on tail exceedance loss
    state_loss_weights: tuple[float, float, float] = (1.5, 1.5, 3.0)
    include_coarse_training_inputs: bool = False  # Include coarse trajectory in FiLM training inputs
    gate_init_bias: float = -2.0        # Initialize gated residual near zero
    tail_exceedance_temperature: float = 0.05  # Smooth indicator for tail
    context_dropout_prob: float = 0.25  # Train FiLM runs to tolerate ensemble-free inference
    state_quantile_calibration: bool = True  # Training-only distribution calibration for all states
    quantile_grid_size: int = 401

    # Test inference
    inference_window_overlap: float = 0.5  # Sliding window overlap ratio
    coarse_context_window: int = 101  # Rolling coarse-only window for FiLM test context

    # Metadata compatibility fields; unused by the residual correctors.
    test_sequence_length: int = 128
    latent_dim: int = 8
    kl_weight: float = 0.0
    dt_override: float | None = None
    tail_loss_interval: int = 1

    @property
    def sigma_value(self) -> float:
        """Extract sigma from dataset_npz filename, e.g., 'dataset_tau8.00_sigma1e-03'."""
        match = re.search(r'sigma([\de\-+\.]+)', self.dataset_npz)
        if match:
            return float(match.group(1))
        return 0.0

    def output_root(self) -> Path:
        """Return the shared ML_Results root under the archive root."""
        if self.results_dir is not None:
            return Path(self.results_dir).resolve()
        return Path(self.dataset_npz).resolve().parents[1] / "ML_Results"

    def output_stem(self) -> str:
        """Return the result folder stem without a repeat suffix."""
        sigma_str = f"{self.sigma_value:.2e}".replace("+", "")
        model_str = self.model_kind[:3].lower()  # "det" or "fil"
        cond_str = "nofilm" if self.n_ensemble_members_used == 0 else (
            f"vec{self.n_eigcov_modes}" if self.eigencov else "meanstd"
        )
        arch_str = self.model_architecture.lower()
        train_time_str = f"{self.train_time}"
        return (
            f"sigma{sigma_str}_{model_str}_{arch_str}_ens{self.n_ensemble_members_used}_"
            f"{cond_str}_stride{self.ml_stride}_T{train_time_str}_run{self.run_id_train}_{self.run_id_test}"
        )

    def output_dir_for_repeat(self, repeat_index: int) -> Path:
        """Return the output directory for a specific repeat index."""
        return self.output_root() / f"{self.output_stem()}_rep{int(repeat_index):03d}"

    @property
    def output_dir(self) -> Path:
        """Return a deterministic output directory, without allocating a new repeat."""
        if self.repeat_index is None:
            return self.output_root() / self.output_stem()
        return self.output_dir_for_repeat(self.repeat_index)

    def resolve_output_dir(self) -> Path:
        """Return a non-overwriting output directory for a training run."""
        if self.repeat_index is not None:
            return self.output_dir_for_repeat(self.repeat_index)
        if not self.auto_increment_repeats:
            return self.output_root() / self.output_stem()

        for repeat in range(10000):
            candidate = self.output_dir_for_repeat(repeat)
            if not candidate.exists():
                return candidate

        raise RuntimeError("Could not allocate a free repeat output directory.")

    @property
    def base_dt(self) -> float:
        return 0.003

    @property
    def ml_dt(self) -> float:
        return self.base_dt * max(int(self.ml_stride), 1)

    @property
    def train_interval_length(self) -> int:
        """Convert train_time (seconds) to timesteps in the saved data."""
        return int(self.train_time / self.base_dt)

    @property
    def test_interval_length(self) -> int:
        """Convert test_time (seconds) to timesteps in the saved data."""
        return int(self.test_time / self.base_dt)
