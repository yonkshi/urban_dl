#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4bele
#SBATCH --constrain "khazadum|rivendell|belegost|shire"
#SBATCH --mail-type ALL
#SBATCH --mail-user pyshi@kth.se
#SBATCH --output /Midgard/home/%u/run_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/run_logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate project_env
python train.py --cuda --learning-rate ${LEARNING_RATE} --epochs ${EPOCHS}




#
. ~/miniconda3/etc/profile.d/conda.sh
conda activate gpu2
python train_xview2.py -c ${CONFIG_NAME}
