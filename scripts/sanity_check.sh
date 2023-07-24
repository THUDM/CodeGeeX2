#!/bin/bash
# This script is used to check the correctness of code generation benchmarks.

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

# enviroment settings
DATASET=humanevalx
GENERATION_MODE=completion
N_CPU_WORKERS=16
TIMEOUT=5

# Check HumanEval-X
for l in python java js cpp go rust;
do
    LANGUAGE=$l
    echo "Evaluating $l"
    DATA_DIR=$MAIN_DIR/benchmark/$DATASET/
    DATA_PATH=$DATA_DIR/$DATASET\_$LANGUAGE.jsonl.gz
    OUTPUT_PATH=$MAIN_DIR/output/$DATASET/$LANGUAGE
        
    JOB_ID=sanity-check-$LANGUAGE
    mkdir -p "$OUTPUT_PATH/$JOB_ID"

    # evaluation settings
    EVAL_INPUT_PATH=$DATA_PATH
    EVAL_OUTPUT_PATH=$OUTPUT_PATH/$JOB_ID
    
    if [ $LANGUAGE = rust ]; then
        TIMEOUT=300
        echo "Setting timeout to $TIMEOUT for Rust"
    fi

    RUN_CMD="python \
        $MAIN_DIR/evaluation/evaluation.py \
        --test_groundtruth=True \
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

    eval "$RUN_CMD"
done
