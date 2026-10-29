#!/bin/bash

MODE="gacl"
DATASET="cifar100" 
# DATASET="imagenet-r"
# DATASET="tinyimagenet"

N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=$SLURM_ARRAY_TASK_ID

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
    BUFFER=5000; GAMMA=100
elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
    BUFFER=5000; GAMMA=100
elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
    BUFFER=5000; GAMMA=100
else
    echo "Undefined setting"
    exit 1
fi

NOTE="GACL" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

for seed in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES="3" python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir local_datasets \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM --buffer_size $BUFFER  --gamma_main $GAMMA 
done
wait
