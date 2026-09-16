#!/bin/bash

# Submit one GPU job per task. A task is one (feature section, evaluation
# method) pair, and its folds run inside that job -- 5 sections x 3 methods
# = 15 jobs, 105 GP fits. The task count is read from `gpc tasks` rather than
# hardcoded, so it follows configs/default.json.
#
# Every setting lives in the block below; nothing is exported before submitting.
# cpus-per-task matches "threads" in configs/default.json -- change both together.

DATE=$(date +%Y%m%d)
run_tag="rxnpredict_v1"
conda_env="/usr/local/usrapps/ddomlab/kagoble/gp_collab_hazel_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs="runs/${run_tag}"

output_root="/share/ddomlab/kagoble/working_space/gp_collab_rxnpredict/HPC_history/hpc_${DATE}"
output_dir="${output_root}/${run_tag}"
mkdir -p "$output_dir" || exit 1

gpu_request="gpu:l40:1"
walltime="03:59:00"
# Ten epochs into a throwaway run folder first, to size walltime and memory.
pilot=0
# Chain collection behind the training jobs. Set to 0 to collect by hand later.
submit_collection=1

if [[ "$pilot" == "1" ]]; then
    runs="runs/${run_tag}_pilot"
    walltime="00:59:00"
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

job_ids=()
for task_id in $(seq 0 $((n_tasks - 1))); do
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
