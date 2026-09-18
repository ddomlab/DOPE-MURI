#!/bin/bash

# Submit one job per task. A task is one (feature section, evaluation method)
# pair, and its folds run inside that job -- 1 section x 3 methods = 3 jobs,
# 15 GP fits (5 LOLO + 5 + 5). The task count is read from
# `gpc tasks` rather than hardcoded, so it follows configs/default.json.
#
# Every setting lives in the block below; nothing is exported before submitting.
# cpus-per-task matches "threads" in configs/default.json -- change both together.

DATE=$(date +%Y%m%d)
run_tag="cernak_v1"
conda_env="/usr/local/usrapps/ddomlab/kagoble/gp_collab_hazel_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs="runs/${run_tag}"

output_root="${project_root}/HPC_history/hpc_${DATE}"
output_dir="${output_root}/${run_tag}"
mkdir -p "$output_dir" || exit 1

gpu_request="gpu:l40:1"
walltime="01:00:00"
# Ten epochs into a throwaway run folder first, to size walltime and memory.
pilot=0
# Chain collection behind the training jobs. Set to 0 to collect by hand later.
submit_collection=1

if [[ "$pilot" == "1" ]]; then
    runs="runs/${run_tag}_pilot"
    walltime="00:20:00"
    epochs="--epochs 10"
    submit_collection=0
else
    epochs=""
fi

# Read the registry with the installed environment's interpreter, so the
# script works from a bare login shell with no conda env activated.
env_python="${conda_env}/bin/python"
if [[ ! -x "$env_python" ]]; then
    echo "No interpreter at ${env_python}. Run hazel/setup_environment.sh first." >&2
    exit 1
fi
n_tasks=$(cd "$project_root" && "$env_python" -m gpc tasks | tail -n +2 | wc -l)
if [[ -z "$n_tasks" || "$n_tasks" -eq 0 ]]; then
    echo "Could not read the task registry from ${project_root}." >&2
    exit 1
fi
echo "Submitting ${n_tasks} tasks to ${runs}"

# Per-task walltime. run.model_overrides[<model>].walltime wins for that model; every
# other task keeps the default above; model_overrides is empty by default here.
# This screen is small -- 1320 wells, 34 one-hot inputs, so at most a 1105x1105
# Cholesky per fit against 3955 rows x 160 inputs for the rxnpredict sweep -- so
# the default below is deliberately loose. Every task is 5 folds. Run
# with pilot=1 once and size this from the measured time before trusting it.
default_walltime="${walltime}"
declare -A model_walltime
while IFS=$'	' read -r _m _w; do
    [[ -n "$_m" ]] && model_walltime["$_m"]="$_w"
done < <(cd "$project_root" && "$env_python" -c "
import json
cfg = json.load(open('configs/default.json'))
for m, o in cfg.get('run', {}).get('model_overrides', {}).items():
    if o.get('walltime'):
        print(m + chr(9) + o['walltime'])
")
mapfile -t task_model < <(cd "$project_root" && "$env_python" -m gpc tasks | tail -n +2 | awk '{print $2}')

job_ids=()
for task_id in $(seq 0 $((n_tasks - 1))); do
    walltime="${model_walltime[${task_model[$task_id]}]:-$default_walltime}"
    job_id=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=${walltime}
#SBATCH --partition=gpu
#SBATCH --account=ddomlab_gpu
#SBATCH --gres=${gpu_request}
#SBATCH --mem=16G
#SBATCH --job-name="gpc_${DATE}_t${task_id}"
#SBATCH --output="${output_dir}/task_${task_id}_GPU.out"
#SBATCH --error="${output_dir}/task_${task_id}_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0

cd ${project_root}
"${conda_env}/bin/python" -m gpc --runs ${runs} train --task-id ${task_id} ${epochs}
EOT
)
    job_ids+=("$job_id")
done

echo "Submitted: ${job_ids[*]}"

if [[ "$submit_collection" == "1" ]]; then
    dependency=$(IFS=:; echo "${job_ids[*]}")
    sbatch --dependency="afterok:${dependency}" <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=compute
#SBATCH --account=ddomlab_cpu
#SBATCH --mem=8G
#SBATCH --job-name="gpc_collect_${DATE}"
#SBATCH --output="${output_dir}/collect.out"
#SBATCH --error="${output_dir}/collect.err"

source ~/.bashrc

cd ${project_root}
"${conda_env}/bin/python" -m gpc --runs ${runs} collect
EOT
    echo "Collection chained behind the training jobs."
fi

echo "Logs: ${output_dir}"
