#!/bin/bash

# Check the installed environment on an allocated GPU before any training.
# Reports driver visibility, the torch CUDA build, the GPU name, and a
# float64 Cholesky, which is the operation exact GP fitting depends on.
# Run this after setup_environment.sh and before the pilot.

DATE=$(date +%Y%m%d)
model="hazel_gp"
conda_env="/usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

output_root="/share/ddomlab/kagoble/working_space/GP_collab/results/HPC_history/hpc_${DATE}"
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
#SBATCH --job-name="hazel_gp_preflight_${DATE}"
#SBATCH --output="${output_dir}/${model}_preflight_GPU.out"
#SBATCH --error="${output_dir}/${model}_preflight_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0
conda activate ${conda_env}

nvidia-smi

cd ${project_root}
python -m hazel_gp check-env --device cuda
EOT

echo "Preflight submitted. Logs: ${output_dir}"
