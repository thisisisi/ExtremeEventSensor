#!/usr/bin/env python3
"""Generate a LowDim dataset ready for training."""

import argparse
from pathlib import Path

from data_generation import DatasetConfig, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=510.0, help="simulation duration")
    parser.add_argument("--time-steps", type=int, default=170_000, help="reference time points")
    parser.add_argument("--coarse-factor", type=int, default=1, help="coarse-grid spacing multiplier")
    parser.add_argument("--save-stride", type=int, default=5, help="save every Nth integrated point") # reduces storage and speeds up training; must be >= 1
    parser.add_argument("--tau", type=float, default=8.0, help="nudging relaxation time")
    parser.add_argument("--sigma", type=float, default=1e-6, help="ensemble noise amplitude")
    parser.add_argument("--runs", type=int, default=2, help="independent trajectories") # used to index multiple runs in the dataset; must be >= 1
    parser.add_argument("--members", type=int, default=16, help="ensemble members per run")
    parser.add_argument("--seed", type=int, default=0, help="random seed") # sets the seed for reproducible ensemble generation
    parser.add_argument("--output", type=Path, help="output directory; default is derived from tau and sigma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        t_span=(0.0, args.duration),
        n_ref=args.time_steps,
        coarse_factor=args.coarse_factor,
        save_stride=args.save_stride,
        tau=args.tau,
        sigma=args.sigma,
        n_runs=args.runs,
        ensemble_size=args.members,
    )
    output = generate_dataset(config, output_dir=args.output, seed=args.seed)
    print(f"Dataset written to {output}")


if __name__ == "__main__":
    main()
