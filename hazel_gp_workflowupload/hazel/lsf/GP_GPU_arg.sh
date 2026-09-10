#!/bin/bash

# Submit one GPU job per task in the prepared input bundle's task registry.
# The registry is model-major: every representation gets the same split list,
# so task ids run 0..n_tasks-1 across all representations. Nothing is
# discovered automatically apart from the task count, which is read from the
# bundle manifest that `pack-inputs` verified.
#
# Core count matches the "threads": 4 setting in configs/default.json;
# change both together.

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

job_name="hazel_gp_gpu_${DATE}"
# Chain collection after every training job reports done. Set to 0 for a
# partial or selected submission, then submit GP_collect_CPU_arg.sh by hand.
submit_collection=1

n_tasks=$(sed -n 's/.*"n_tasks"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "${project_root}/${bundle}/manifest.json")
if [[ -z "$n_tasks" ]]; then
    echo "No n_tasks in ${project_root}/${bundle}/manifest.json. Extract the upload bundle first." >&2
    exit 1
fi

for task_id in $(seq 0 $((n_tasks - 1))); do
    bsub <<EOT
#BSUB -n 4
#BSUB -W 1:59
#BSUB -q short_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[a10 || a30 || a100 || l40 || h100]"
#BSUB -J "${job_name}"
#BSUB -o "${output_dir}/${model}_task${task_id}_GPU.out"
#BSUB -e "${output_dir}/${model}_task${task_id}_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0
conda activate ${conda_env}

cd ${project_root}
python -m hazel_gp run --bundle "${bundle}" \
                       --runs "${runs}" \
                       --task-id ${task_id} \
                       --device cuda
EOT
done

echo "Submitted ${n_tasks} GPU training jobs as ${job_name}."

if [[ "$submit_collection" == 1 ]]; then
    bsub <<EOT
#BSUB -n 1
#BSUB -W 0:20
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=4GB]"
#BSUB -w "done(${job_name})"
#BSUB -J "hazel_gp_collect_${DATE}"
#BSUB -o "${output_dir}/${model}_collect.out"
#BSUB -e "${output_dir}/${model}_collect.err"

source ~/.bashrc
conda activate ${conda_env}

cd ${project_root}
mkdir -p exports
python -m hazel_gp collect --bundle "${bundle}" --runs "${runs}" && \
python -m hazel_gp pack-results --runs "${runs}" \
                                --output "exports/results_\${LSB_JOBID}.zip"
EOT
    echo "Collection queued behind ${job_name}."
else
    echo "Automatic collection disabled. Submit GP_collect_CPU_arg.sh once the training jobs finish."
fi
