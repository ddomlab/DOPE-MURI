#!/bin/bash

# Collect a finished (or partial) run into summary/predictions/lolo tables.
# Submit this by hand when GP_GPU_arg.sh was run with submit_collection=0,
# or to re-collect after retrying individual tasks.

DATE=$(date +%Y%m%d)
run_tag="rxnpredict_pcs_v3"
conda_env="/usr/local/usrapps/ddomlab/kagoble/gp_collab_hazel_py312"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs="runs/${run_tag}"

output_root="${project_root}/HPC_history/hpc_${DATE}"
output_dir="${output_root}/${run_tag}"
mkdir -p "$output_dir" || exit 1

sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=compute
#SBATCH --account=ddomlab_cpu
#SBATCH --mem=8G
#SBATCH --job-name="gpc_collect_${DATE}"
#SBATCH --output="${output_dir}/collect_${run_tag}.out"
#SBATCH --error="${output_dir}/collect_${run_tag}.err"

source ~/.bashrc

cd ${project_root}
"${conda_env}/bin/python" -m gpc --runs ${runs} collect
EOT

echo "Collection submitted. Logs: ${output_dir}"
