# Submitting uploaded zips on Hazel

Upload the zips from this folder to `/share/ddomlab/kagoble`, open a terminal
there, and run the block below. Before running it, set `TAGS` to the uploaded
zips' names without `.zip`.

The conda environment is `/usr/local/usrapps/ddomlab/kagoble/gp_collab_hazel_py312`.
Zips built before 2026-10-01 point at the nonexistent `gp_collab_hub_py312`, and
the `sed` line repoints them. On newer zips it changes nothing.

```bash
TAGS="reizman_summit_20261001 ahneman_doyle_dft_v2_20261001"

cd /share/ddomlab/kagoble
for d in $TAGS; do
  unzip -q $d.zip
  sed -i 's#gp_collab_hub_py312#gp_collab_hazel_py312#g' $d/hazel/setup_environment.sh $d/hazel/slurm/*.sh $d/configs/*.json $d/RUN_ON_HPC.md
done
for d in $TAGS; do
  cd /share/ddomlab/kagoble/$d && bash hazel/slurm/GP_GPU_arg.sh
done
squeue -u $USER
```

Each zip submits one GPU job per (feature section, method), with a collection
job chained behind them.

## After the jobs finish

```bash
sacct -u $USER -S today --format=JobName%40,State,Elapsed,Start
```

Bring back `/share/ddomlab/kagoble/<tag>/runs/<tag>/` into `gp_collab_hub/runs/`.
Logs for failed or timed-out jobs are in `/share/ddomlab/kagoble/<tag>/HPC_history/`.

## Used on 2026-10-01

```bash
cd /share/ddomlab/kagoble
unzip -q reizman_summit_20261001.zip
unzip -q ahneman_doyle_dft_v2_20261001.zip
for d in reizman_summit_20261001 ahneman_doyle_dft_v2_20261001; do
  sed -i 's#gp_collab_hub_py312#gp_collab_hazel_py312#g' $d/hazel/setup_environment.sh $d/hazel/slurm/*.sh $d/configs/*.json $d/RUN_ON_HPC.md
done
cd /share/ddomlab/kagoble/reizman_summit_20261001 && bash hazel/slurm/GP_GPU_arg.sh
cd /share/ddomlab/kagoble/ahneman_doyle_dft_v2_20261001 && bash hazel/slurm/GP_GPU_arg.sh
squeue -u $USER
```
