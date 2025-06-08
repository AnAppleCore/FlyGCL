#!/bin/bash

#SBATCH -J New_L2P_iblurry_cifar100_N70_M10
#SBATCH -p batch_agi
#SBATCH  -w agi2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH -t 7-0
#SBATCH -o %x_%j.log

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=$SLURM_NNODES

conda --version
python --version

# CIL CONFIG
METHOD="L2P_V0_"
CHECKPOINT="R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1_0"
META_PATH="/data/sunguanglong/meta_vit2/R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1/meta_epoch_0.pth"
# COR_PATH="/data/sunguanglong/meta_vit2/R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1/cov_matrix_backbone5.npy"

# CHECKPOINT="R100_noW_ibot_sample_200_lr_0.1_numtask_50_steps_1_0"
# META_PATH="/data/sunguanglong/meta_vit2/R100_noW_ibot_sample_200_lr_0.1_numtask_50_steps_1/meta_epoch_0.pth"
# COR_PATH="/data/sunguanglong/meta_vit2/R100_noW_ibot_sample_200_lr_0.1_numtask_50_steps_1/cov_matrix_backbone30.npy"

# CHECKPOINT="R100_noW_sup1k_sample_200_lr_0.1_numtask_50_steps_1_2"
# META_PATH="/data/sunguanglong/meta_vit2/R100_noW_sup1k_sample_200_lr_0.1_numtask_50_steps_1/meta_epoch_2.pth"
# COR_PATH="/data/sunguanglong/meta_vit2/R100_noW_sup1k_sample_200_lr_0.1_numtask_50_steps_1/cov_matrix_backbone2.npy"

MODE="L2P"

DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet

NOTE=${METHOD}_${DATASET}_${CHECKPOINT}
N_TASKS=5
N=70
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=1

OPT="adam"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="L2P" EVAL_PERIOD=1000
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="L2P" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256
    DATA_DIR="/data/datasets/CIFAR"

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="L2P" EVAL_PERIOD=1000
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="L2P" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256
    DATA_DIR="/data/datasets/imagenet-r"

elif [ "$DATASET" == "cub200" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="L2P" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256
    DATA_DIR="/data/datasets/CUB200_2011"

elif [ "$DATASET" == "imagenet" ]; then
    N_TASKS=10 MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="L2P" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=0.03 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    BATCHSIZE=256; LR=0.05 OPT_NAME=$OPT SCHED_NAME="multistep" MEMORY_EPOCH=100

else
    echo "Undefined setting"
    exit 1
fi

echo "Batch size $BATCHSIZE  onlin iter $ONLINE_ITER"
for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir $DATA_DIR \
    --note $NOTE --eval_period $EVAL_PERIOD --memory_epoch $MEMORY_EPOCH --n_worker 2 \
    --meta_path $META_PATH > /home/sunguanglong/DGIL/MISA/results/output/vit_others/$NOTE.txt 2>&1
done
