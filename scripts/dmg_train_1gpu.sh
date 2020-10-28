#!/usr/bin/env bash

# Submit a job array without a physical .sbatch file using config files a HERE document.
# https://en.wikipedia.org/wiki/Here_document
# https://slurm.schedmd.com/job_array.html
#
# Before submitting prepare a `queue` folder where each file corresponds to one config.
# Each file is called `array.<date>.<id>.yaml`. Files corresponding to succesful runs
# are deleted. If the run fails the config file is moved to an `error` folder.
#
# Variables and commands in the HERE document work like this:
# - ${RUNS_PATH}     is evaluated *now* and takes the value
#                    from the current shell (as defined below),
#                    it's useful to pass paths and thyperparameters
# - \${SLURM_JOB_ID} is evaluated when the job starts, therefore
#                    you can access variables set by slurm
# - $(date)          is evaluated *now* and takes the value
#                    from the current shell (as defined above)
# - \$(date)         is evaluated when the job starts, therefore
#                    you can run commands on the node

LOG_PATH="${HOME}/slurm_logs/urban_dl/"
CONFIG_NAME=$1
USER=pyshi
PFX=/local_storage/users/pshi/topo_trajs_data
SLURM_ARRAY_TASK_ID=0

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${LOG_PATH}/%x_%J.out"
#SBATCH --error="${LOG_PATH}/%x_%J.err"
#SBATCH --mail-type=FAIL
#SBATCH --constrain="gondor|shire|khazadum|rivendell|belegost"
#SBATCH --mail-user="pyshi@kth.se"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --job-name=${CONFIG_NAME}
# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "JOB NAME: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo ""
nvidia-smi
# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate gpu3
cd ../
# Train and save the exit code of the python script
python damage_train.py -c ${CONFIG_NAME}
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE