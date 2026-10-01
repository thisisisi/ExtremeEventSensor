"""Experiment definitions and stable artifact identifiers."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str
    module: str
    artifact_id: str

    def load(self):
        """Import only this implementation, preserving Keras registration order."""
        return import_module(self.module)


EXPERIMENTS = {
    experiment.name: experiment
    for experiment in (
        Experiment('baseline', 'Baseline', 'baseline', 'S0'),
        Experiment('pooled', 'Pooled ensemble', 'ablation_pooled', 'S1'),
        Experiment('film', 'FiLM', 'film', 'S4'),
        Experiment('random_context', 'Random context', 'ablation_random_context', 'S10'),
        Experiment('coarse_statistics', 'State statistics', 'ablation_coarse_statistics', 'S11'),
        Experiment('ensemble_mean', 'Ensemble mean', 'ablation_ensemble_mean', 'S12'),
        Experiment('ensemble_variance', 'Ensemble variance', 'ablation_ensemble_variance', 'S13'),
        Experiment('covariance_concat', 'Covariance concatenation', 'ablation_covariance_concat', 'S14'),
        Experiment('mean_spread_concat', 'Mean/spread concatenation', 'ablation_mean_spread_concat', 'S15'),
    )
}


def get_experiment(value: str) -> Experiment:
    """Resolve a descriptive name or a saved artifact's identifier."""
    key = str(value).lower()
    if key in EXPERIMENTS:
        return EXPERIMENTS[key]
    for experiment in EXPERIMENTS.values():
        if experiment.artifact_id == str(value).upper():
            return experiment
    raise ValueError(f'Unsupported experiment {value!r}; choose {", ".join(EXPERIMENTS)}')


def experiment_name(value: str) -> str:
    return get_experiment(value).name


def artifact_id(value: str) -> str:
    return get_experiment(value).artifact_id


def load_experiment(strategy):
    """Load a train/evaluate implementation; None selects the unconditioned model."""
    return import_module('unconditioned') if strategy is None else get_experiment(strategy).load()
