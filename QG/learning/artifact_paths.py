"""Checkpoint/result filenames, independent of numerical and plotting libraries."""
import os
import json
import hashlib

from experiments import artifact_id


def _sanitize_tokens(*toks):
    return [str(t).replace(' ', '').replace('/', '_') for t in toks if t is not None and str(t) != '']

def _repeat_suffix(repeat_index):
    if repeat_index is None:
        return ''
    return f"_rep{int(repeat_index):03d}"

def _extra_token_parts(extra_tokens):
    if not extra_tokens:
        return []
    if isinstance(extra_tokens, dict):
        items = sorted(extra_tokens.items())
    else:
        items = list(extra_tokens)

    parts = []
    for key, value in items:
        if value is None or str(value) == '':
            continue
        parts.extend(_sanitize_tokens(f"{key}{value}"))
    return parts

def model_save_path(neural_nets_dir, case_name, model_type, basis_type, Nepochs, dt_ml, T, j_ens, *, repeat_index=None):
    return os.path.join(
            neural_nets_dir,
            f"{case_name}_{model_type}_{basis_type}_epochs_{Nepochs}_dt_{dt_ml}_T_{T}_j{j_ens}{_repeat_suffix(repeat_index)}.h5"
        )

def results_save_path(ml_results_dir, case_name, model_type, basis_type, Nepochs, dt_ml, T, Nens, *, repeat_index=None):
    return os.path.join(ml_results_dir,
        f"{case_name}_{model_type}_{basis_type}_loss_L2_epochs_{Nepochs}_dt_{dt_ml}_T_{T}_Nens_{Nens}_Ensemble_Stats_Big{_repeat_suffix(repeat_index)}.mat"
    )

def model_save_path2(neural_nets_dir, case_name, model_type, basis_type,
                     Nepochs, dt_ml, T, j_ens, *, strategy=None, n_members=None, extra_tokens=None, repeat_index=None):
    if strategy is None:
        return model_save_path(
            neural_nets_dir, case_name, model_type, basis_type, Nepochs, dt_ml, T, j_ens,
            repeat_index=repeat_index,
        )
    strategy = artifact_id(strategy)
    sig = {
        "case": case_name, "model": model_type, "basis": basis_type,
        "Nepochs": int(Nepochs), "dt_ml": float(dt_ml), "T": int(T), "j": int(j_ens),
        "strategy": str(strategy).upper(), "k": int(n_members) if n_members is not None else None,
        "repeat_index": int(repeat_index) if repeat_index is not None else None,
        "extra": (extra_tokens or {}),
    }
    h8 = hashlib.md5(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    toks = [f"S{str(strategy).upper()}"]
    if n_members is not None: toks.append(f"k{int(n_members)}")
    toks.extend(_extra_token_parts(extra_tokens))
    suffix = _repeat_suffix(repeat_index)
    fname = (f"{case_name}_{model_type}_{basis_type}_epochs_{Nepochs}_dt_{dt_ml}_T_{T}_j{j_ens}"
             f"{'_' + '-'.join(toks) if toks else ''}{suffix}.h5")
    return os.path.join(neural_nets_dir, fname)

def results_save_path2(ml_results_dir, case_name, model_type, basis_type,
                       Nepochs, dt_ml, T, Nens, *, strategy=None, n_members=None, extra_tokens=None, repeat_index=None):
    if strategy is None:
        return results_save_path(
            ml_results_dir, case_name, model_type, basis_type, Nepochs, dt_ml, T, Nens,
            repeat_index=repeat_index,
        )
    strategy = artifact_id(strategy)
    sig = {
        "case": case_name, "model": model_type, "basis": basis_type,
        "Nepochs": int(Nepochs), "dt_ml": float(dt_ml), "T": int(T), "Nens": int(Nens),
        "strategy": str(strategy).upper(), "k": int(n_members) if n_members is not None else None,
        "repeat_index": int(repeat_index) if repeat_index is not None else None,
        "extra": (extra_tokens or {}),
    }
    h8 = hashlib.md5(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    base = f"{case_name}_{model_type}_{basis_type}_loss_L2_epochs_{Nepochs}_dt_{dt_ml}_T_{T}_Nens_{Nens}"
    toks = [f"S{str(strategy).upper()}"]
    if n_members is not None: toks.append(f"k{int(n_members)}")
    toks.extend(_extra_token_parts(extra_tokens))
    suffix = _repeat_suffix(repeat_index)
    fname = f"{base}{'_' + '-'.join(toks) if toks else ''}{suffix}_Ensemble_Stats_Big.mat"
    return os.path.join(ml_results_dir, fname)
