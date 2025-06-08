#!/bin/bash

#SBATCH -J DP_Siblurry_CIFAR_500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH -t 6-0
#SBATCH -o %x_%j_%a.log
#SBATCH -e %x_%j_%a.err

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

conda --version
python --version

# CIL CONFIG
METHOD="MISA_gcl"

# CHECKPOINT="R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1_5"
# META_PATH="/data//meta_vit2/R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1/meta_epoch_5.pth"
# COR_PATH="/data//meta_vit2/R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1/cov_matrix_backbone5.npy"

# CHECKPOINT="R100_noW_ibot_sample_200_lr_0.1_numtask_50_steps_1_30"
# META_PATH="/data//meta_vit2/R100_noW_ibot_sample_200_lr_0.1_numtask_50_steps_1/meta_epoch_30.pth"
# COR_PATH="/data//meta_vit2/R100_noW_ibot_sample_200_lr_0.1_numtask_50_steps_1/cov_matrix_backbone30.npy"

CHECKPOINT="R100_noW_sup1k_sample_200_lr_0.1_numtask_50_steps_1_2"
META_PATH="/data//meta_vit2/R100_noW_sup1k_sample_200_lr_0.1_numtask_50_steps_1/meta_epoch_2.pth"
COR_PATH="/data//meta_vit2/R100_noW_sup1k_sample_200_lr_0.1_numtask_50_steps_1/cov_matrix_backbone2.npy"

# LINEAR_PATH="/data//meta_vit2/R100_noW_sup_sample_200_lr_0.1_numtask_50_steps_1/backbone5_linear1000_meta_epoch_0.pth"
MODE="DualPrompt"
DATASET="imagenet-r" # cifar10, cifar100, tinyimagenet, imagenet, imagenet-r, nch, cub200, cars196
COR_COEF=0.0
NOTE=${METHOD}_coef_${COR_COEF}_21k_${DATASET}_${CHECKPOINT}
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=1
# echo "SEEEDS="$SEEDS

OPT="adam"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256
    DATA_DIR="/data/datasets/CIFAR"

elif [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=500 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    DATA_DIR="/data/datasets/imagenet-r"

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=15000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "nch" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "cub200" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    DATA_DIR="/data/datasets/CUB200_2011"

elif [ "$DATASET" == "cars196" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    DATA_DIR="/data/datasets/CARS196/split_imgs"

elif [ "$DATASET" == "cub175" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "gtsrb" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=2000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "wikiart" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

else
    echo "Undefined setting"
    exit 1
fi

echo "Batch size $BATCHSIZE  onlin iter $ONLINE_ITER"
for RND_SEED in $SEEDS
do
    python -W ignore main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir $DATA_DIR \
    --note $NOTE --eval_period $EVAL_PERIOD --transforms autoaug --memory_epoch $MEMORY_EPOCH --n_worker 8 --rnd_NM \
    --load_pt \
    --cor_path $COR_PATH --pretrain_cor --cor_coef $COR_COEF \
    --meta_path $META_PATH  > /home//DGIL/MISA/results/output/vit_gcl/$NOTE.txt 2>&1
done

