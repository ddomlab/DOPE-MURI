#!/bin/bash

# Collect a finished run and pack the results for download. Submit this by
# hand when a training script had submit_collection=0, or to re-collect.
#
# Collection re-verifies test row identities, observed values, metadata and
# output hashes against the bundle, so it runs on a compute node with the
# training environment rather than on the login node.
#
# A full collection rejects an incomplete run on purpose. Set allow_partial=1
# to report a pilot or a partially finished run; the report then carries
# incomplete flags and is not comparable to a full benchmark.

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

allow_partial=0
# Checkpoints and preprocessors make the archive much larger. Include them
# only to re-collect locally or to reload trained models.
include_checkpoints=0

partial_flag=""
if [[ "$allow_partial" == 1 ]]; then partial_flag="--allow-partial"; fi
checkpoint_flag=""
if [[ "$include_checkpoints" == 1 ]]; then checkpoint_flag="--include-checkpoints"; fi

mkdir -p "${project_root}/exports" || exit 1

bsub <<EOT
#BSUB -n 1
#BSUB -W 0:20
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=4GB]"
#BSUB -J "hazel_gp_collect_${DATE}"
#BSUB -o "${output_dir}/${model}_collect_manual.out"
#BSUB -e "${output_dir}/${model}_collect_manual.err"

source ~/.bashrc
conda activate ${conda_env}

cd ${project_root}
python -m hazel_gp collect --bundle "${bundle}" \
                           --runs "${runs}" ${partial_flag} && \
python -m hazel_gp pack-results --runs "${runs}" \
                                --output "exports/results_${run_tag}_\${LSB_JOBID}.zip" ${checkpoint_flag}
EOT

echo "Collection submitted for ${runs} (allow_partial=${allow_partial})."
echo "The archive lands in exports/ under a unique job id, so earlier downloads are kept."
