#!/usr/bin/env bash
# Author: Isabella Thiel <thieli@mit.edu>
# Common environment, configuration snapshot, and runtime logging for QG jobs.

start_job() {
    module purge
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
    if [[ ! -x "$PYTHON_BIN" ]]; then
      echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
      exit 1
    fi

    # Keep concurrent CPU jobs from each creating a full-node thread pool.
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export TF_CPP_MIN_LOG_LEVEL=2


    RUN_STARTED_EPOCH="$(date +%s)"
    RUN_STARTED_ISO="$(date -Iseconds)"
    echo "Job started at: $RUN_STARTED_ISO"
    echo "Running on node: $(hostname)"
    echo "Job ID: ${SLURM_JOB_ID:-manual}"
    echo "Python: $PYTHON_BIN"
    mkdir -p runs
    CONFIG_SRC="${CONFIG_SRC:-ml_config.yaml}"
    if [[ ! -f "$CONFIG_SRC" ]]; then
        echo "ERROR: Config not found: $CONFIG_SRC" >&2
        exit 1
    fi
    JOBTAG="${SLURM_JOB_ID:-manual}"
    CONFIG_DST="runs/ml_config.${JOBTAG}.yaml"
    cp -f "$CONFIG_SRC" "$CONFIG_DST"
    echo "Using config copy: $CONFIG_DST"
}

finish_job() {
    RUN_ENDED_EPOCH="$(date +%s)"
    RUN_ENDED_ISO="$(date -Iseconds)"
    ELAPSED_SECONDS="$((RUN_ENDED_EPOCH - RUN_STARTED_EPOCH))"

    SACCT_STATE="NA"
    SACCT_ELAPSED="NA"
    SACCT_REQ_MEM="NA"
    SACCT_MAX_RSS="NA"
    if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
      sacct_rows="$(sacct -j "$SLURM_JOB_ID" --format=State,Elapsed,ReqMem,MaxRSS -P -n 2>/dev/null || true)"
      if [[ -n "$sacct_rows" ]]; then
        sacct_compact="$(printf "%s\n" "$sacct_rows" | awk -F'|' '
          function toMB(v) {
            if (v == "" || v == "Unknown" || v == "0") return -1
            if (v ~ /K$/) { sub(/K$/, "", v); return v / 1024.0 }
            if (v ~ /M$/) { sub(/M$/, "", v); return v + 0 }
            if (v ~ /G$/) { sub(/G$/, "", v); return (v + 0) * 1024.0 }
            if (v ~ /T$/) { sub(/T$/, "", v); return (v + 0) * 1024.0 * 1024.0 }
            return v + 0
          }
          {
            if (state == "" && $1 != "") state = $1
            if (elapsed == "" && $2 != "") elapsed = $2
            if (req == "" && $3 != "") req = $3
            mb = toMB($4)
            if ($4 != "" && (maxrss == "" || mb > maxmb)) { maxrss = $4; maxmb = mb }
          }
          END {
            if (state == "") state = "NA"
            if (elapsed == "") elapsed = "NA"
            if (req == "") req = "NA"
            if (maxrss == "") maxrss = "NA"
            printf "%s|%s|%s|%s", state, elapsed, req, maxrss
          }
        ' )"
        IFS='|' read -r SACCT_STATE SACCT_ELAPSED SACCT_REQ_MEM SACCT_MAX_RSS <<< "$sacct_compact"
      fi
    fi

    # Optional combined sweep runtime log (set by main.py sweep).
    if [[ -n "${SWEEP_LOG_CSV:-}" ]]; then
      mkdir -p "$(dirname "$SWEEP_LOG_CSV")"
      if [[ ! -f "$SWEEP_LOG_CSV" ]]; then
        echo "submitted_at,slurm_job_id,mode,repeat_index,seed,sig,n_ens,Ntime,strategy,started_at,ended_at,elapsed_seconds,sacct_state,sacct_elapsed,sacct_req_mem,max_rss" > "$SWEEP_LOG_CSV"
      fi

      csv_line="${SWEEP_SUBMITTED_AT:-},${SLURM_JOB_ID:-},${SWEEP_MODE:-},${SWEEP_REPEAT_INDEX:-},${SWEEP_SEED:-},${SWEEP_SIG:-},${SWEEP_N_ENS:-},${SWEEP_NTIME:-},${SWEEP_STRATEGY:-},${RUN_STARTED_ISO},${RUN_ENDED_ISO},${ELAPSED_SECONDS},${SACCT_STATE},${SACCT_ELAPSED},${SACCT_REQ_MEM},${SACCT_MAX_RSS}"
      if command -v flock >/dev/null 2>&1; then
        {
          flock -x 200
          echo "$csv_line" >> "$SWEEP_LOG_CSV"
        } 200>"${SWEEP_LOG_CSV}.lock"
      else
        echo "$csv_line" >> "$SWEEP_LOG_CSV"
      fi

      echo "Appended runtime row to: $SWEEP_LOG_CSV"
    fi


    # Print completion information
    echo
    echo "Job completed at: $(date)"
}
