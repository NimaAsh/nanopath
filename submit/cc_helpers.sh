# Interactive shell helpers for driving nanopath runs on Compute Canada / Alliance clusters.
# Version-controlled so Rorqual / Trillium / Nibi stay in sync via `git pull` (Alliance gives each
# cluster its own /home, so a per-.bashrc copy would otherwise drift). To use, add ONE line to the
# .bashrc on each cluster (and drop any older inline copy of these functions):
#     source ~/nanopath/submit/cc_helpers.sh
# Optional env overrides: NANOPATH_REPO (default ~/nanopath), NANOPATH_RUN_BASE (default the
# per-cluster /scratch/$USER/nanopath/main-leopard). All paths are per-cluster, so the same file
# works everywhere unchanged.

# Last N of your jobs over the past 14 days (batch steps collapsed with -X).
lastjobs() {
    local n=${1:-5}
    local out=$(sacct -u $USER -X -S now-14days --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU)
    echo "$out" | head -n 2
    echo "$out" | tail -n "$n"
}

# Print the probe-score breakdown for a run (index into nprun, or an explicit run dir).
# Prefers labless_submission.json (canonical metric names); else maps summary.json's
# final_probe_* slots to the same names so local, unsubmitted runs show all 8 slots too.
npscores() {
    local arg=${1:-1}
    local run_dir
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        run_dir=$(nprun "$arg")
    else
        run_dir="$arg"
    fi

    echo "RUN_DIR=$run_dir"
    python - "$run_dir" <<'PY'
import json, pathlib, sys

run = pathlib.Path(sys.argv[1])
p = run / "labless_submission.json"
if p.exists():
    metrics = json.load(open(p))["run"]["metrics"]
else:
    s = json.load(open(run / "summary.json"))
    # summary.json names the 8 slots final_probe_<task>_<metric>; map to the labless canonical names.
    m = {"linear_mean_f1": "linear", "knn_mean_f1": "knn", "fewshot_mean_f1": "few_shot",
         "seg_mean_jaccard": "seg_jaccard", "slide_mean_auc": "progression_auc", "auc_mean": "mutation_auc",
         "survival_mean_cindex": "survival_cindex", "robustness_mean": "robustness"}
    metrics = {m.get(k.removeprefix("final_probe_"), k.removeprefix("final_probe_")): v
               for k, v in s.items() if k.startswith("final_probe_")}
    metrics["mean_probe_score"] = s.get("final_probe_score")

order = ["mean_probe_score", "linear", "knn", "few_shot", "seg_jaccard", "progression_auc", "mutation_auc", "survival_cindex", "robustness"]
for k in order:
    if k in metrics and isinstance(metrics[k], (int, float)):
        print(f"{k}: {metrics[k]:.4f}")
PY
}

# Dry-run the labless submitter against a run and print its scores (no upload).
lablessdryrun() {
    local n=${1:-1}
    local repo=${NANOPATH_REPO:-$HOME/nanopath}
    local run_dir=$(nprun "$n")
    local run_name=${2:-$(basename "$run_dir")}

    echo "RUN_DIR=$run_dir"
    cd "$repo" || return
    source .venv/bin/activate

    ./labless/submit_to_labless.py output_dir="$run_dir" run_name="$run_name" notes="dry run" dry_run=true
    local status=$?

    echo
    echo "Scores:"
    npscores "$run_dir"

    return "$status"
}

# Nth most recently finished run dir (by summary.json mtime) under the per-cluster run base.
nprun() {
    local n=${1:-1}
    local base=${2:-${NANOPATH_RUN_BASE:-/scratch/$USER/nanopath/main-leopard}}
    find "$base" -name summary.json -type f -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr | sed -n "${n}p" | cut -d' ' -f2-
}

# Overview of the last N finished runs: index (for nprun/lablessdryrun/npsubmit), score, run dir.
nplist() {
    local n=${1:-10} i d
    for i in $(seq 1 "$n"); do
        d=$(nprun "$i"); [[ -n "$d" ]] || continue
        printf "%2d  %s  %s\n" "$i" "$(python - "$d/summary.json" 2>/dev/null <<'PY'
import json,sys
try: print("%.4f"%json.load(open(sys.argv[1]))["final_probe_score"])
except Exception: print("  ?   ")
PY
)" "$d"
    done
}

# Submit the Nth most recent run to Labless. run_name defaults to the dir name (truncated to 20 chars);
# pass run_name + notes to override. Validate first with `lablessdryrun N` (non-destructive). 20 subs/24h.
npsubmit() {
    local n=${1:-1}
    local repo=${NANOPATH_REPO:-$HOME/nanopath}
    local d; d=$(nprun "$n")
    local run_name=${2:-$(basename "$d" | cut -c1-20)}
    local notes=${3:-"$(basename "$d")"}
    echo "RUN_DIR=$d  run_name=$run_name"
    cd "$repo" || return
    source .venv/bin/activate
    ./labless/submit_to_labless.py output_dir="$d" run_name="$run_name" notes="$notes"
}

# wandb sync the offline run matching a finished run dir's wandb id.
wsyncnp() {
    local arg=${1:-1}
    local repo=${NANOPATH_REPO:-$HOME/nanopath}
    local run_dir
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        run_dir=$(nprun "$arg")
    else
        run_dir="$arg"
    fi

    local id
    id=$(python - "$run_dir/summary.json" <<'PY'
import json, sys
print((json.load(open(sys.argv[1])).get("wandb") or {}).get("id") or "")
PY
)

    local wb_dir
    wb_dir=$(find /scratch/$USER/nanopath/wandb -type d -name "*${id}" -print -quit)

    echo "RUN_DIR=$run_dir"
    echo "WANDB_DIR=$wb_dir"
    cd "$repo" || return
    source .venv/bin/activate
    wandb sync "$wb_dir"
}

# Your RUNNING/PENDING training jobs. Matches both launcher job names: the unified
# cc_train_1gpu.sbatch sets --job-name=nanopath; the retired per-cluster launchers used main_1gpu.
nprunjobs() {
    squeue -u "$USER" --noheader --sort=-i --format="%i %j %T %M %R" \
        | awk '($2=="main_1gpu" || $2=="nanopath") && ($3=="RUNNING" || $3=="PENDING"){print}'
}

# tail -f the .out (or .err) log of the Nth RUNNING/PENDING training job.
nplogrun() {
    local n=${1:-1}
    local kind=${2:-out}
    local job
    job=$(nprunjobs | sed -n "${n}p" | awk '{print $1}')

    if [[ -z "$job" ]]; then
        echo "No RUNNING/PENDING job at index $n."
        nprunjobs
        return 1
    fi

    echo "JOB=$job"
    tail -f "/scratch/$USER/nanopath-${job}.${kind}"
}

# Bundle the last N finished runs (summary + metrics + source + logs + wandb) into a tgz for download.
nppackruns() {
    local n=${1:-1}
    local out=${2:-/scratch/$USER/nanopath/nanopath-last-${n}-runs-$(date +%Y%m%d-%H%M%S).tgz}
    local stage
    stage=$(mktemp -d /scratch/$USER/nanopath/export.XXXXXX)

    mkdir -p "$stage/runs" "$stage/logs" "$stage/wandb"

    for i in $(seq 1 "$n"); do
        local run_dir
        run_dir=$(nprun "$i")
        [[ -n "$run_dir" ]] || continue

        local dest="$stage/runs/${i}-$(basename "$run_dir")"
        mkdir -p "$dest"

        echo "Packing $i: $run_dir"

        cp -a "$run_dir/summary.json" "$dest/"
        cp -a "$run_dir/metrics.jsonl" "$dest/"
        [[ -f "$run_dir/labless_submission.json" ]] && cp -a "$run_dir/labless_submission.json" "$dest/"
        [[ -d "$run_dir/labless_source" ]] && rsync -a "$run_dir/labless_source/" "$dest/labless_source/"
        [[ -d "$run_dir/thunder" ]] && rsync -a --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' "$run_dir/thunder/" "$dest/thunder/"

        local job_id
        job_id=$(python - "$run_dir/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("slurm_job_id") or "")
PY
)
        [[ -n "$job_id" && -f "/scratch/$USER/nanopath-${job_id}.out" ]] && cp -a "/scratch/$USER/nanopath-${job_id}.out" "$stage/logs/"
        [[ -n "$job_id" && -f "/scratch/$USER/nanopath-${job_id}.err" ]] && cp -a "/scratch/$USER/nanopath-${job_id}.err" "$stage/logs/"

        local wandb_id
        wandb_id=$(python - "$run_dir/summary.json" <<'PY'
import json, sys
print((json.load(open(sys.argv[1])).get("wandb") or {}).get("id") or "")
PY
)
        local wb_dir
        wb_dir=$(find /scratch/$USER/nanopath/wandb -type d -name "*${wandb_id}" -print -quit)
        [[ -n "$wb_dir" ]] && rsync -a "$wb_dir" "$stage/wandb/"
    done

    tar -C "$(dirname "$stage")" -czf "$out" "$(basename "$stage")"
    echo "$out"
}
