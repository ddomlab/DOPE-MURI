#!/bin/bash

# Check the installed environment on an allocated GPU before any training.
# Reports driver visibility, the torch CUDA build and a float64 Cholesky,
# which is the operation exact GP fitting depends on.

DATE=$(date +%Y%m%d)
conda_env="/usr/local/usrapps/ddomlab/kagoble/gp_collab_hazel_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

output_root="/share/ddomlab/kagoble/working_space/gp_collab_hazel/HPC_history/hpc_${DATE}"
output_dir="${output_root}/preflight"
mkdir -p "$output_dir" || exit 1

gpu_request="gpu:a100:1"

sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=${gpu_request}
#SBATCH --mem=4G
#SBATCH --job-name="gpc_preflight_${DATE}"
#SBATCH --output="${output_dir}/preflight_GPU.out"
#SBATCH --error="${output_dir}/preflight_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0
conda activate ${conda_env}

nvidia-smi

cd ${project_root}
python -m gpc check-env
python -m gpc tasks
python -m gpc splits
EOT

echo "Preflight submitted. Logs: ${output_dir}"
