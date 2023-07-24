#!/bin/bash
# This script is used to generate solutions of HumanEval-X.

# Examples (MODE=(gen, eval, both)):
# MODE=gen bash ./scripts/run_humanevalx.sh

if [ -z "$MODE" ]
then
  MODE="both"
fi

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

# enviroment settings
HOSTLIST=$SCRIPT_DIR/hostlist
WORLD_SIZE=1
DATASET=humanevalx
GENERATION_MODE=completion
MODEL_NAME=codegeex2-6b
MODEL_PATH=/pathto/codegeex2-6b/
N_CPU_WORKERS=16
TIMEOUT=5

# generation settings
## pass@1 greedy
NUM_SAMPLES=1
MICRO_BSZ=1
TEMP=1.0
TOPK=1
TOPP=1.0
MAX_LENGTH=1024
SEED=42
GREEDY=1

## pass@1 estimated
# NUM_SAMPLES=20
# MICRO_BSZ=1
# TEMP=0.2
# TOPK=0
# TOPP=0.95
# MAX_LENGTH=1024
# SEED=42
# GREEDY=0

## pass@10 & pass@100
# NUM_SAMPLES=200
# MICRO_BSZ=4
# TEMP=0.8
# TOPK=0
# TOPP=0.95
# MAX_LENGTH=1024
# SEED=42
# GREEDY=0

for l in python java js cpp go rust;
do
    LANGUAGE=$l
    DATA_DIR=$MAIN_DIR/benchmark/$DATASET/
    DATA_PATH=$DATA_DIR/$DATASET\_$LANGUAGE.jsonl.gz
    OUTPUT_PATH=$MAIN_DIR/output/$DATASET/$LANGUAGE
    TODAY=$(date +%y%m%d)
    CHANNEL_PORT=$(expr $RANDOM + 5000)
    MASTER_PORT=$(expr $RANDOM + 8000)
    JOB_ID=$MODEL_NAME-$LANGUAGE-greedy$GREEDY-ns$NUM_SAMPLES-t$TEMP-topp$TOPP-seed$SEED
    mkdir -p "$OUTPUT_PATH/$JOB_ID"

    # evaluation settings
    EVAL_INPUT_PATH=$OUTPUT_PATH/$JOB_ID
    EVAL_OUTPUT_PATH=$OUTPUT_PATH/$JOB_ID

    # nccl options
    OPTIONS_NCCL="export NCCL_DEBUG=warn; export NCCL_IB_DISABLE=0; export NCCL_IB_GID_INDEX=3"
    OPTIONS_PATH="export PATH=$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    CWD=$(pwd)

    gen_func() {
        echo "Generating......"
        # set master ip for zmq server
        if [ -z "$HOSTLIST" ]; then
            ZMQ_ADDR=$(hostname -i)
            echo "$ZMQ_ADDR" > "./hostfile"
            HOSTLIST="./hostfile"
        else
            ZMQ_ADDR=$(cat $HOSTLIST | head -n 1)
        fi
        echo "master_ip: $ZMQ_ADDR"

        # run generation
        RUN_CMD="python \
            $MAIN_DIR/evaluation/generation.py \
            --hostfile $HOSTLIST \
            --channel-ip $ZMQ_ADDR \
            --channel-port $CHANNEL_PORT \
            --master-port $MASTER_PORT \
            --model-path $MODEL_PATH \
            --temperature $TEMP \
            --top-p $TOPP \
            --top-k $TOPK \
            --greedy $GREEDY \
            --max-length $MAX_LENGTH \
            --micro-batch-size $MICRO_BSZ \
            --samples-per-problem $NUM_SAMPLES \
            --model-name $MODEL_NAME \
            --dataset-type $DATASET \
            --language-type $LANGUAGE \
            --generation-mode $GENERATION_MODE \
            --data-path $DATA_PATH \
            --output-path $OUTPUT_PATH/$JOB_ID \
            --log-path $OUTPUT_PATH/$JOB_ID/$TODAY-generation.log \
            --gen-node-world-size $WORLD_SIZE \
            --seed $SEED"

        RUN_CMD="$OPTIONS_NCCL; $OPTIONS_PATH; $RUN_CMD"
        RUN_CMD="cd $CWD; $RUN_CMD"

        if (( WORLD_SIZE != 1 )); then
            RUN_CMD="pdsh -R ssh -w ^$HOSTLIST \"$RUN_CMD\""
        fi

        eval "$RUN_CMD"
    }

    eval_func() {
        echo "Evaluating......"

        if [ $LANGUAGE = rust ]; then
            TIMEOUT=300
            echo "Setting timeout to $TIMEOUT for Rust"
        fi
        RUN_CMD="python \
            $MAIN_DIR/evaluation/evaluation.py \
            --input_path $EVAL_INPUT_PATH \
            --output_path $EVAL_OUTPUT_PATH \
            --log-path $OUTPUT_PATH/$JOB_ID/$TODAY-evaluation.log \
            --model_name $MODEL_NAME \
            --language_type $LANGUAGE \
            --dataset_type $DATASET \
            --generation_mode $GENERATION_MODE \
            --n_workers $N_CPU_WORKERS \
            --tmp_dir $MAIN_DIR/benchmark/$DATASET/$LANGUAGE \
            --problem_file $DATA_PATH \
            --timeout $TIMEOUT"

        # inspecting results
        INSPECT_CMD="python \
            $MAIN_DIR/evaluation/inspect_jsonl.py \
            --data_path $EVAL_OUTPUT_PATH/result-$JOB_ID.jsonl \
            --log-path $OUTPUT_PATH/$JOB_ID/$TODAY-inspect.txt"

        eval "$RUN_CMD && $INSPECT_CMD"
    }

    case $MODE in
    "gen")
        gen_func
        ;;
    "eval")
        eval_func
        ;;
    "both")
        gen_func
        eval_func
        ;;
    *)
        echo "Unsupported MODE (gen, eval, both): $MODE"
        exit 1
        ;;
    esac
done
