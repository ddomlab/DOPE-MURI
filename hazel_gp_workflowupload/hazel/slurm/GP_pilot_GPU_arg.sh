#!/bin/bash

# Pilot: one full-size task per representation, capped at a few optimizer
# steps, written to its own runs directory. Use it to measure fit time,
# memory and warnings before committing walltime to the full registry.
#
# The pilot ids are the first task of each representation in the registry,
# read from tasks.csv rather than hardcoded, because the split count per
# representation changes whenever the bundle is re-prepared. To pilot a
# specific set instead, list the ids in pilot_task_ids below.

DATE=$(date +%Y%m%d)
model="hazel_gp"
run_tag="pilot_v1"
bundle="inputs"
runs="runs/${run_tag}"
max_steps=10
conda_env="/usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

output_root="/share/ddomlab/kagoble/working_space/GP_collab/results/HPC_history/hpc_${DATE}"
output_dir="${output_root}/${run_tag}"
mkdir -p "$output_dir" || exit 1

job_name="hazel_gp_pilot_${DATE}"
gpu_request="gpu:a100:1"
# A short partner pilot uses --partition=gpu_partners with --qos=short_gpu,
# which requires matching account privileges.
partition="gpu"

# Leave empty for one task per representation, or set explicitly, e.g.
# pilot_task_ids=(0 54 108 162 216)
pilot_task_ids=()

if [[ ${#pilot_task_ids[@]} -eq 0 ]]; then
    pilot_tasks=($(awk -F, 'NR>1 && !seen[$2]++ {print $1 "|" $2}' "${project_root}/${bundle}/tasks.csv"))
else
    pilot_tasks=()
    for task_id in "${pilot_task_ids[@]}"; do
        pilot_tasks+=("${task_id}|selected")
    done
fi

if [[ ${#pilot_tasks[@]} -eq 0 ]]; then
    echo "No tasks found in ${project_root}/${bundle}/tasks.csv." >&2
    exit 1
fi

for pilot_task in "${pilot_tasks[@]}"; do
    IFS='|' read -r task_id representation <<< "$pilot_task"

    sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:59:00
#SBATCH --partition=${partition}
#SBATCH --gres=${gpu_request}
#SBATCH --mem=16G
#SBATCH --job-name="${job_name}"
#SBATCH --output="${output_dir}/${model}_${representation}_task${task_id}_GPU.out"
#SBATCH --error="${output_dir}/${model}_${representation}_task${task_id}_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0
conda activate ${conda_env}

cd ${project_root}
python -m hazel_gp run --bundle "${bundle}" \
                       --runs "${runs}" \
                       --task-id ${task_id} \
                       --device cuda \
                       --max-steps ${max_steps}
EOT
done

echo "Submitted ${#pilot_tasks[@]} pilot GPU jobs as ${job_name}."
echo "Collection is not chained: a pilot is partial by design."
echo "Review it with GP_collect_CPU_arg.sh after setting allow_partial=1 and run_tag=${run_tag}."
