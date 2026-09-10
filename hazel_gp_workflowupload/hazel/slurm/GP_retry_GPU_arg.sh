#!/bin/bash

# Resubmit the specific tasks that failed or never completed. List them
# explicitly below; nothing is discovered automatically. Find the ids with:
#
#   conda activate /usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312
#   python -m hazel_gp status --bundle inputs --runs runs/experiment_v1
#
# Completed tasks are hash-checked and skipped, so an id that already
# finished costs nothing. A scheduler kill can leave a task's .running lock
# (or .initializing) behind: confirm the old job has ended with squeue, then
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
gpu_request="gpu:a100:1"
submit_collection=1

# Replace these with the actual failed/missing task ids.
retry_task_ids=(7 19 88)

if [[ ${#retry_task_ids[@]} -eq 0 ]]; then
    echo "retry_task_ids is empty. Add the ids reported by hazel_gp status." >&2
    exit 1
fi

job_ids=()
for task_id in "${retry_task_ids[@]}"; do
    job_id=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:59:00
#SBATCH --partition=gpu
#SBATCH --gres=${gpu_request}
#SBATCH --mem=16G
#SBATCH --job-name="${job_name}"
#SBATCH --output="${output_dir}/${model}_retry_task${task_id}_GPU.out"
#SBATCH --error="${output_dir}/${model}_retry_task${task_id}_GPU.err"

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
)
    job_ids+=("${job_id%%;*}")
done

echo "Submitted ${#job_ids[@]} retry GPU jobs as ${job_name}."

if [[ "$submit_collection" == 1 ]]; then
    dependency="afterok:$(IFS=:; echo "${job_ids[*]}")"
    sbatch --dependency="$dependency" <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=compute
#SBATCH --mem=4G
#SBATCH --job-name="hazel_gp_collect_${DATE}"
#SBATCH --output="${output_dir}/${model}_collect_retry.out"
#SBATCH --error="${output_dir}/${model}_collect_retry.err"

source ~/.bashrc
conda activate ${conda_env}

cd ${project_root}
mkdir -p exports
python -m hazel_gp collect --bundle "${bundle}" --runs "${runs}" && \
python -m hazel_gp pack-results --runs "${runs}" \
                                --output "exports/results_\${SLURM_JOB_ID}.zip"
EOT
    echo "Collection queued behind ${job_name}."
fi
