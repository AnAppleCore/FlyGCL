#!/bin/bash


date
# seeds=(1 21 42 3473 10741 32450 93462 85015 64648 71950 87557 99668 55552 4811 10741)
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
# export WORLD_SIZE=$SLURM_NNODES
export WORLD_SIZE=1


conda --version
python --version

# CIL CONFIG
NOTE="MVP" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="mvp"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet, imagenet-r
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=1 # "1 2 3", just separate seeds by spaces
echo "SEEEDS="$SEEDS

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=0 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "nch" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "gtsrb" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "wikiart" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100


else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    echo "RND_SEED="$RND_SEED
    python -W ignore main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --seeds $SEEDS \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /services/scratch/thoth/zkang/  \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 8 --transforms autoaug --rnd_NM \
    --gamma 2.0 \
    --use_mcr \
    --use_afs \
    --use_mask \
    --use_contrastiv \

done