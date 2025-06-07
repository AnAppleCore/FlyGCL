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

NOTE="ISA" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

MODE="fam" 
DATASET="imagenet900" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=10
N=101
M=0
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=1 # "1 2 3", just separate seeds by spaces
echo "SEEEDS="$SEEDS

OPT="adam"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=1
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=64; TEMPBATCHSIZE=65 LR=5e-3 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=256 NUM_EPOCHS=3

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=64; LR=5e-3 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=256

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=1
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=24; TEMPBATCHSIZE=65 LR=5e-4 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=100 NUM_EPOCHS=3

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=256; TEMPBATCHSIZE=65 LR=1e-4 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=100 NUM_EPOCHS=1

elif [ "$DATASET" == "imagenet900" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="CodaPrompt" EVAL_PERIOD=-1
    BATCHSIZE=256; TEMPBATCHSIZE=65 LR=1e-4 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=100 NUM_EPOCHS=3

elif [ "$DATASET" == "imagenetsub" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=256; TEMPBATCHSIZE=65 LR=1e-4 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=100 NUM_EPOCHS=3

elif [ "$DATASET" == "imagenet100" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=256; TEMPBATCHSIZE=65 LR=1e-4 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=100 NUM_EPOCHS=5

elif [ "$DATASET" == "cub175" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="DualPrompt" EVAL_PERIOD=-1
    BATCHSIZE=256; TEMPBATCHSIZE=65 LR=1e-5 OPT_NAME="fam" SCHED_NAME="fam" MEMORY_EPOCH=100 NUM_EPOCHS=5

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
    --seeds $SEEDS \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --temp_batchsize $TEMPBATCHSIZE  \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /local_datasets \
    --note $NOTE --eval_period $EVAL_PERIOD --transforms autoaug --memory_epoch $MEMORY_EPOCH --n_worker 8 --rnd_NM \
    --num_epochs $NUM_EPOCHS --nobatchmask --isa --e_proj --e_proj 
done
