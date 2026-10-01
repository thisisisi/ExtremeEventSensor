"""Shared configuration for the QG comparison jobs."""
from dataclasses import dataclass
from pathlib import Path
import sys

# Appending keeps this directory's modules ahead of the learning implementation.
sys.path.append(str(Path(__file__).resolve().parents[1] / 'learning'))
from experiments import EXPERIMENTS, get_experiment


@dataclass(frozen=True)
class ComparisonConfig:
    sigmas: tuple[float, ...] = (0.01,)
    ensemble_sizes: tuple[int, ...] = (2,)
    training_durations: tuple[int, ...] = (50,)
    unconditioned_durations: tuple[int, ...] = (50, 1000)
    experiments: tuple[str, ...] = tuple(EXPERIMENTS)
    model_type: str = 'STORN'
    basis_type: str = 'rnudge16.0'
    epochs: int = 2000
    dt_ml: float = 0.1


CONFIG = ComparisonConfig()
EXPERIMENT_IDS = tuple(get_experiment(name).artifact_id for name in CONFIG.experiments)
DISPLAY_NAMES = {get_experiment(name).artifact_id: get_experiment(name).label for name in CONFIG.experiments}
