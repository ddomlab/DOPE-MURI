#!/bin/bash

# Run once on a Hazel login node to install the environment into usrapps.
# Keep conda_env here in step with conda_env in the submission scripts.
#
# Compute nodes cannot write to usrapps and home has a small quota, so the
# conda and pip caches go to /share instead.

set -euo pipefail

conda_env="/usr/local/usrapps/ddomlab/kagoble/gp_collab_hazel_py312"
cache_dir="/share/ddomlab/kagoble/conda_cache"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -e "$conda_env" ]]; then
    echo "Environment destination already exists. Activate it, or choose a new prefix." >&2
    exit 1
fi

module load conda
eval "$(conda shell.bash hook)"

mkdir -p "$cache_dir"
export CONDA_PKGS_DIRS="${cache_dir}/conda_pkgs"
export PIP_CACHE_DIR="${cache_dir}/pip"

conda create --prefix "$conda_env" python=3.12 pip --yes
conda activate "$conda_env"

cd "$project_root"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > "${cache_dir}/gp_collab_hub_environment_freeze.txt"

echo "Environment installed at ${conda_env}."
echo "Next: bash hazel/slurm/GP_preflight_GPU_arg.sh"
