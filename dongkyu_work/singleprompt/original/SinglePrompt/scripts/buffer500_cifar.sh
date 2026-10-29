MODE="Buffer"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet, imagenet-r, nch, cub200
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3 4 5" # "1 2 3", just separate seeds by spaces
POS_PROMPT=5
PROMPT_LENGTH=20
LOGIT_TYPE=cos_sim
MEM_SIZE=500
echo "SEEDS="$SEEDS

if [ "$DATASET" == "cifar100" ]; then
    ONLINE_ITER=3
    EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
elif [ "$DATASET" == "imagenet-r" ]; then
    ONLINE_ITER=3
    EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
elif [ "$DATASET" == "tinyimagenet" ]; then
    ONLINE_ITER=3
    EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
else
    echo "Undefined setting"
    exit 1
fi

echo "Batch size $BATCHSIZE  online iter $ONLINE_ITER"
for RND_SEED in $SEEDS
do

    NOTE="${DATASET}_pos${POS_PROMPT}_len${PROMPT_LENGTH}_buffer${MEM_SIZE}_logitType${LOGIT_TYPE}"

    echo "Running with SEED=$RND_SEED"

    python -W ignore main.py --mode $MODE \
        --dataset $DATASET \
        --n_tasks $N_TASKS --m $M --n $N \
        --seeds $RND_SEED \
        --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        --lr $LR --batchsize $BATCHSIZE \
        $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./local_dataset \
        --note $NOTE --eval_period $EVAL_PERIOD --transforms autoaug --n_worker 8 --rnd_NM \
        --pos_prompt $POS_PROMPT --prompt_length $PROMPT_LENGTH --logit_type $LOGIT_TYPE --memory_size $MEM_SIZE

done