#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=24
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --job-name=open-diffusion
ml purge
source source /p/project/laionize/miniconda/bin/activate open_clip
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR
export PYTHONPATH="$PYTHONPATH:$PWD/src"
#export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_CACHE=cache
export TRANSFORMERS_OFFLINE=1
srun --cpu_bind=none,v --accel-bind=gn python -u train.py config=$1
