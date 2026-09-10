#!/bin/bash

# CPU fallback for the same task registry as GP_GPU_arg.sh. The packaged
# CUDA-enabled PyTorch wheel runs on CPU unchanged. Core count matches the
# "threads": 4 setting in configs/default.json; change both together.

DATE=$(date +%Y%m%d)
model="hazel_gp"
run_tag="experiment_v1_cpu"
bundle="inputs"
runs="runs/${run_tag}"
conda_env="/usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

output_root="/share/ddomlab/kagoble/working_space/GP_collab/results/HPC_history/hpc_${DATE}"
output_dir="${output_root}/${run_tag}"
mkdir -p "$output_dir" || exit 1

job_name="hazel_gp_cpu_${DATE}"
submit_collection=1

n_tasks=$(sed -n 's/.*"n_tasks"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "${project_root}/${bundle}/manifest.json")
if [[ -z "$n_tasks" ]]; then
    echo "No n_tasks in ${project_root}/${bundle}/manifest.json. Extract the upload bundle first." >&2
    exit 1
fi

job_ids=()
for task_id in $(seq 0 $((n_tasks - 1))); do
    job_id=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH --job-name="${job_name}"
#SBATCH --output="${output_dir}/${model}_task${task_id}_CPU.out"
#SBATCH --error="${output_dir}/${model}_task${task_id}_CPU.err"

source ~/.bashrc
conda activate ${conda_env}

cd ${project_root}
python -m hazel_gp run --bundle "${bundle}" \
                       --runs "${runs}" \
                       --task-id ${task_id} \
                       --device cpu
EOT
)
    job_ids+=("${job_id%%;*}")
done

echo "Submitted ${#job_ids[@]} CPU training jobs as ${job_name}."

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
#SBATCH --output="${output_dir}/${model}_collect.out"
#SBATCH --error="${output_dir}/${model}_collect.err"

source ~/.bashrc
conda activate ${conda_env}

cd ${project_root}
mkdir -p exports
python -m hazel_gp collect --bundle "${bundle}" --runs "${runs}" && \
python -m hazel_gp pack-results --runs "${runs}" \
                                --output "exports/results_\${SLURM_JOB_ID}.zip"
EOT
    echo "Collection queued behind ${job_name}."
else
    echo "Automatic collection disabled. Submit GP_collect_CPU_arg.sh once the training jobs finish."
fi
