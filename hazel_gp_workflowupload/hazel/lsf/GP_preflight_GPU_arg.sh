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

bsub <<EOT
#BSUB -n 1
#BSUB -W 0:10
#BSUB -q short_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[a10 || a30 || a100 || l40 || h100]"
#BSUB -J "hazel_gp_preflight_${DATE}"
#BSUB -o "${output_dir}/${model}_preflight_GPU.out"
#BSUB -e "${output_dir}/${model}_preflight_GPU.err"

source ~/.bashrc
module load cuda/12.1
module load gcc/9.3.0
conda activate ${conda_env}

nvidia-smi

cd ${project_root}
python -m hazel_gp check-env --device cuda
EOT

echo "Preflight submitted. Logs: ${output_dir}"
