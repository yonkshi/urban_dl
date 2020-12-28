
#!/usr/bin/env bash

# Submit a sbatch job without a physical .sbatch file using a HERE document.
# https://en.wikipedia.org/wiki/Here_document
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
#
# Before submitting you can replace `sbatch` with `cat` to check that
# all variables and commands work as expected, you can also uncomment
# `break 1000` below to break the for loops after the first iteration


#for learning_rate in .001 .01; do
#for weight_decay in .001 .00001; do
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

RUNS_PATH="/Midgard/home/pshi/slurm_logs/urban_dl"
RUN_CONFIG_PREFIX="array.$(date +'%F_%T.%N')"
SOURCE=$1
DEST=$2
SLURM_MAX_TASKS=1
SLURM_ARRAY_TASK_IDD=1
sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%x_%A_%a.out"
#SBATCH --error="${RUNS_PATH}/%x_%A_%a.err"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="$dkerekes@kth.se"
#SBATCH --constrain="khazadum"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=cp
#SBATCH --array=1-${SLURM_ARRAY_TASK_IDD}%${SLURM_MAX_TASKS}

# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi
# Activate conda
cp $SOURCE $DEST
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%x_%A_%a.out"
#SBATCH --error="${RUNS_PATH}/%x_%A_%a.err"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="$dkerekes@kth.se"
#SBATCH --constrain="rivendell"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=cp
#SBATCH --array=1-${SLURM_ARRAY_TASK_IDD}%${SLURM_MAX_TASKS}

# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi
# Activate conda
cp $SOURCE $DEST
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%x_%A_%a.out"
#SBATCH --error="${RUNS_PATH}/%x_%A_%a.err"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="$dkerekes@kth.se"
#SBATCH --constrain="belegost"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=cp
#SBATCH --array=1-${SLURM_ARRAY_TASK_IDD}%${SLURM_MAX_TASKS}

# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi
# Activate conda
cp $SOURCE $DEST
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%x_%A_%a.out"
#SBATCH --error="${RUNS_PATH}/%x_%A_%a.err"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="$dkerekes@kth.se"
#SBATCH --constrain="shire"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=cp
#SBATCH --array=1-${SLURM_ARRAY_TASK_IDD}%${SLURM_MAX_TASKS}

# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi
# Activate conda
cp $SOURCE $DEST
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%x_%A_%a.out"
#SBATCH --error="${RUNS_PATH}/%x_%A_%a.err"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="$dkerekes@kth.se"
#SBATCH --constrain="gondor"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=cp
#SBATCH --array=1-${SLURM_ARRAY_TASK_IDD}%${SLURM_MAX_TASKS}

# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi
# Activate conda
cp $SOURCE $DEST
EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"
HERE