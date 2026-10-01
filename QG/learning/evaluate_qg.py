#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""

import sys
import fcntl
from experiments import load_experiment
from utils import load_config

print('Importing Tensorflow (serialized per node)... ', end='', flush=True)
with open('/tmp/qg_tensorflow_import.lock', 'w') as tf_import_lock:
    fcntl.flock(tf_import_lock, fcntl.LOCK_EX)
    import tensorflow as tf
    fcntl.flock(tf_import_lock, fcntl.LOCK_UN)
print('Done!')
print('Importing Keras... ', end='', flush=True)
print('Done!')

if __name__ == "__main__":

    cfg = load_config()
    test_again = False # retest even if results exist

    data     = cfg.get("data", {}) or {}
    training = cfg.get("training", {}) or {}
    paths    = cfg.get("paths", {}) or {}
    ensemble = cfg.get("ensemble", {}) or {}

    # data
    CASE_NAME_TEST = data.get("case_name_test", None)
    CASE_NAME_TRAIN = data.get("case_name_train", None)
    BASIS_TYPE     = data.get("basis_type", "rnudge16.0")
    DT_ML          = float(data.get("dt_ml", 0.1))
    T              = int(data.get("Ntime_units", 1000))   # test code uses T for NT window
    T_STARTUP    = int(data.get("T_startup", 1000)) # time units to skip at start of
    TEST_TIME_UNITS = int(data.get("test_time_units", 34000))

    # training (used for loading checkpoints & ensemble size)
    MODEL_TYPE = training.get("model_type", "LSTM")
    NEPOCHS    = int(training.get("Nepochs", 2000))
    NENS       = int(training.get("ensemble_index", 1))
    REPEAT_INDEX = training.get("repeat_index", None)
    SEED = training.get("seed", None)

    # paths
    NEURAL_NETS_DIR = paths.get("neural_nets_dir", "../Neural_Nets")
    ML_RESULTS_DIR  = paths.get("ml_results_dir",  "../ML_Results")
    DATA_DIR        = paths.get("data_dir",        "../ML_Data")

    # ensemble switch
    ENSEMBLE = bool(ensemble.get("enabled", False))
    ENSEMBLE_STRATEGY = ensemble.get("strategy", 'pooled')

    if CASE_NAME_TEST is None:
        print("ERROR: case_name_test is not set")
        sys.exit(1)

    experiment = load_experiment(ENSEMBLE_STRATEGY if ENSEMBLE else None)

    experiment.evaluate(
            model_type=MODEL_TYPE,
            case_name=CASE_NAME_TEST,
            Nens=NENS,
            Nepochs=NEPOCHS,
            dt_ml=DT_ML,
            basis_type=BASIS_TYPE,
            T=T,
            neural_nets_dir=NEURAL_NETS_DIR,
            ml_results_dir=ML_RESULTS_DIR,
            data_dir=DATA_DIR,
            t_startup=T_STARTUP,
            test_time_units=TEST_TIME_UNITS,
            case_name_train=CASE_NAME_TRAIN,
            ensemble_cfg=ensemble,
            repeat_index=REPEAT_INDEX,
            seed=SEED,
            test_again=test_again
        )

    sys.exit(0)
