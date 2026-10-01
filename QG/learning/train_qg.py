#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""

import sys
import time
import numpy as np
import os
import fcntl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Importing Tensorflow (serialized per node)... ', end='', flush=True)
# A sweep can place many jobs on one node. Concurrent cold imports from the
# shared conda filesystem can spend tens of minutes in importlib.get_data.
# /tmp is node-local, so this serializes only the import on each node.
with open('/tmp/qg_tensorflow_import.lock', 'w') as tf_import_lock:
    fcntl.flock(tf_import_lock, fcntl.LOCK_EX)
    import tensorflow as tf
    fcntl.flock(tf_import_lock, fcntl.LOCK_UN)
print('Done!')
print('Importing Keras... ', end='', flush=True)
print('Done!')

from experiments import load_experiment
from utils import load_config, seed_everything

if __name__ == "__main__":

    train_again = False #True # retrain even if model exists
    cfg = load_config()

    data = cfg.get("data", {})
    training = cfg.get("training", {}) or {}
    ensemble = cfg.get("ensemble", {}) or {}
    paths = cfg.get("paths", {}) or {}

    # data
    T = int(data.get("Ntime_units", 1000))
    DT_ML = float(data.get("dt_ml", 0.1))
    CASE_NAME_TRAIN = data.get("case_name_train", None)
    BASIS_TYPE = data.get("basis_type", "rnudge16.0")
    T_STARTUP = int(data.get("T_startup", 1000))

    # train
    MODEL_TYPE = training.get("model_type", "LSTM")
    NEPOCHS = int(training.get("Nepochs", 2000))
    J_ENS = int(training.get("ensemble_index", 1))
    LSTM_HIST_POINTS = int(training.get("lstm_hist_points", 100))
    BATCH_SIZE = int(training.get("batch_size", 32))
    LR = float(training.get("learning_rate", 0.001))
    REPEAT_INDEX = training.get("repeat_index", None)
    SEED = training.get("seed", None)

    # paths
    NEURAL_NETS_DIR = paths.get("neural_nets_dir", "../Neural_Nets")
    ML_RESULTS_DIR = paths.get("ml_results_dir", "../ML_Results")
    DATA_DIR = paths.get("data_dir", "../ML_Data")

    # ensemble
    ENSEMBLE = bool(ensemble.get("enabled", False))
    ENSEMBLE_STRATEGY = ensemble.get("strategy", 'pooled')

    if CASE_NAME_TRAIN is None:
        print("ERROR: CASE_NAME_TRAIN is not set")
        sys.exit(1)

    if T < 1000:
        sub_sample = int(np.floor((20 * T) / 1000))
        sub_sample = max(sub_sample, 1)
    else:
        sub_sample = 20

    if SEED is None and REPEAT_INDEX is not None:
        SEED = 1729 + int(REPEAT_INDEX)
    if SEED is not None:
        seed_everything(SEED)

    experiment = load_experiment(ENSEMBLE_STRATEGY if ENSEMBLE else None)

    tstart = time.time()
    experiment.train(
        model_type=MODEL_TYPE,
        case_name=CASE_NAME_TRAIN,
        j_ens=J_ENS,
        Nepochs=NEPOCHS,
        sub_sample=sub_sample,
        lstm_hist_points=LSTM_HIST_POINTS,
        dt_ml=DT_ML,
        basis_type=BASIS_TYPE,
        T=T,
        batch_size=BATCH_SIZE,
        lr=LR,
        t_startup=T_STARTUP,
        # paths
        neural_nets_dir=NEURAL_NETS_DIR,
        ml_results_dir=ML_RESULTS_DIR,
        data_dir=DATA_DIR,
        # ensemble
        ensemble_cfg=ensemble if ENSEMBLE else None,
        repeat_index=REPEAT_INDEX,
        seed=SEED,
        train_again = train_again)

    t_end = time.time()
    print(f"Training completed in {t_end - tstart:.3f} seconds")
    sys.exit(0)
