#!/bin/bash

# Resubmit the specific tasks that failed or never completed. List them
# explicitly below; nothing is discovered automatically. Find the ids with:
#
#   conda activate /usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312
#   python -m hazel_gp status --bundle inputs --runs runs/experiment_v1
#
# Completed tasks are hash-checked and skipped, so an id that already
# finished costs nothing. A scheduler kill can leave a task's .running lock
# (or .initializing) behind: confirm the old job has ended with bjobs, then
# remove only that task's stale lock before resubmitting it here.

DATE=$(date +%Y%m%d)
model="hazel_gp"
run_tag="experiment_v1"
bundle="inputs"
runs="runs/${run_tag}"
conda_env="/usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

output_root="/share/ddomlab/kagoble/working_space/GP_collab/results/HPC_history/hpc_${DATE}"
output_dir="${output_root}/${run_tag}"
mkdir -p "$output_dir" || exit 1

job_name="hazel_gp_retry_${DATE}"
submit_collection=1

# Replace these with the actual failed/missing task ids.
retry_task_ids=(7 19 88)

if [[ ${#retry_task_ids[@]} -eq 0 ]]; then
    echo "retry_task_ids is empty. Add the ids reported by hazel_gp status." >&2
    exit 1
fi

for task_id in "${retry_task_ids[@]}"; do
    bsub <<EOT
#BSUB -n 4
#BSUB -W 1:59
#BSUB -q short_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[a10 || a30 || a100 || l40 || h100]"
#BSUB -J "${job_name}"
#BSUB -o "${output_dir}/${model}_retry_task${task_id}_GPU.out"
#BSUB -e "${output_dir}/${model}_retry_task${task_id}_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0
conda activate ${conda_env}

cd ${project_root}
python -m hazel_gp run --bundle "${bundle}" \
                       --runs "${runs}" \
                       --task-id ${task_id} \
                       --device cuda \
                       --retry
EOT
done

echo "Submitted ${#retry_task_ids[@]} retry GPU jobs as ${job_name}."

if [[ "$submit_collection" == 1 ]]; then
    bsub <<EOT
#BSUB -n 1
#BSUB -W 0:20
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=4GB]"
#BSUB -w "done(${job_name})"
#BSUB -J "hazel_gp_collect_${DATE}"
#BSUB -o "${output_dir}/${model}_collect_retry.out"
#BSUB -e "${output_dir}/${model}_collect_retry.err"

source ~/.bashrc
conda activate ${conda_env}

cd ${project_root}
mkdir -p exports
python -m hazel_gp collect --bundle "${bundle}" --runs "${runs}" && \
python -m hazel_gp pack-results --runs "${runs}" \
                                --output "exports/results_\${LSB_JOBID}.zip"
EOT
    echo "Collection queued behind ${job_name}."
fi
