#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -ex

# Kill vLLM if present
pkill -f 'python -m vllm.entrypoints.openai.api_server' && sleep 15

usage() {
    echo "Usage: $0 --model model --bs batch_size --input_len input_len --output_len output_len"
    echo "Options:"
    echo "  --model, -m             Specify the model, possible choices: [llama2-70b, llama2-7b, llama3-8b-instruct], default: llama2-70b"
    echo "  --bs                    Specify the batch size"
    echo "  --hpu_num               Specify the number of HPUs, default: 8"
    echo "  --eager_mode            Turn On or Off eager mode, choices: [On, Off], default: Off"
    echo "  --load_balancer         Turn On or Off load balancer, choices: [On, Off], default: Off"
    echo "  --fp8                   Enable or Disable fp8/quantization, choices: [On, Off], default: Off"
    echo "  --input_len             Specify the size of prompt sequence bucket"
    echo "  --output_len            Specify the size of output sequence"
    echo "  --output_dir, -o        Specify the output dir for logs if RESULT_DIR is not set, default: ./results"
    echo "  --help                  Display this help message"
    exit 1
}

wait_for_server() {
    local port="$1"
    local model="$2"

    timeout=10800
    step=10
    current_time=0

    while [ "$current_time" -lt "$timeout" ]; do
        output=$(curl -s http://localhost:$port/v1/models | grep $model | wc -l)
        if (( $output > 0 )); then
            echo "vLLM server on port $port started"
            return 0
        fi
        sleep $step
        current_time=$((current_time + step))
    done

    echo "vLLM server on port $port didn't start"
    return -1
}

model="mixtral-8x7b"
hpu_num=1
eager_mode="Off"
load_balancer="Off"
delay_sampling="On"
fp8="Off"
batch_size=100
block_size=128
input_len=1024
output_len=1024
max_model_len=2048

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            model=$2
            shift 2
            ;;
        --bs)
            batch_size=$2
            shift 2
            ;;
        --output_dir|-o)
            output_dir=$2
            shift 2
            ;;
        --hpu_num)
            hpu_num=$2
            shift 2
            ;;
        --eager_mode)
            eager_mode=$2
            shift 2
            ;;
        --load_balancer)
            load_balancer=$2
            shift 2
            ;;
        --delay_sampling)
           delay_sampling=$2
            shift 2
            ;;
        --fp8)
            fp8=$2
            shift 2
            ;;
        --input_len)
            input_len=$2
            shift 2
            ;;
        --profile)
            export VLLM_PROFILER_ENABLED=true
            shift 1
            ;;
        --help)
            usage
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

if [[ -n $HELP || -z $model || -z $batch_size ]]; then
    usage
fi

selected_model=$model

case $model in
    "llama3-8b-instruct")
    model="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B-Instruct/"
    ;;
    "llama3-70b-instruct")
    model="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-70B-Instruct/"
    ;;
    "llama3.1-8b-instruct")
    model="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
    ;;
    "llama3.1-70b-instruct")
    model="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-70B-Instruct/"
    ;;
    "llama3.1-405b-instruct")
    model="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-405B-Instruct/"
    ;;
    "mixtral-8x7b")
    model="/root/ckpt/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1/"
    ;;
esac


if [[ $eager_mode == "On" ]]; then
    EAGER_FLAG="--enforce-eager"
else
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
fi

if [[ $delay_sampling == "On" ]]; then
    DELAY_SAMPLING_FLAG="--enable-delayed-sampling  --num-lookahead-slots 1  --use-v2-block-manager"
else
    DELAY_SAMPLING_FLAG=""
fi

if [[ $fp8 == "On" ]]; then
    QUANT_FLAGS="--quantization hqt --kv-cache-dtype hf8 --weights-load-device cpu"
    case $selected_model in
        "llama3-8b-instruct")
        export QUANT_CONFIG=hqt/llama3-8b/quantization_config/maxabs_quant.json
        ;;
        "llama3-70b-instruct")
        QUANT_FLAGS="--quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
        export QUANT_CONFIG=hqt/llama3-70b-8x/quantization_config/maxabs_quant.json
        ;;
        "llama3.1-8b-instruct")
        QUANT_FLAGS="--quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
        export QUANT_CONFIG=hqt/llama3.1-8b-1x/quantization_config/maxabs_quant.json
        ;;
        "llama3.1-70b-instruct")
        QUANT_FLAGS="--quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
        export QUANT_CONFIG=hqt/llama3.1-70b-8x/quantization_config/maxabs_quant.json
        ;;
        "mixtral-8x7b")
        export QUANT_CONFIG=hqt/mixtral-8x7b/quantization_config/maxabs_quant.json
        ;;
    esac
fi


script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
output_dir=${output_dir:-$script_dir/results}
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="http://localhost:8084/v1"
export EXPERIMENTAL_WEIGHT_SHARING=0
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

export VLLM_GRAPH_RESERVED_MEM=0.3

export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=2

export VLLM_PROMPT_SEQ_BUCKET_MIN=$input_len
export VLLM_PROMPT_SEQ_BUCKET_STEP=$input_len
export VLLM_PROMPT_SEQ_BUCKET_MAX=$input_len

export VLLM_DECODE_BS_BUCKET_MIN=$batch_size
export VLLM_DECODE_BS_BUCKET_STEP=$batch_size
export VLLM_DECODE_BS_BUCKET_MAX=$batch_size

block_bucket_min=$((batch_size * input_len/128))
block_bucket_max=$((batch_size * (input_len+output_len)/128))
export VLLM_DECODE_BLOCK_BUCKET_MIN=$block_bucket_min
export VLLM_DECODE_BLOCK_BUCKET_STEP=4
export VLLM_DECODE_BLOCK_BUCKET_MAX=$block_bucket_max

AVAILABLE_HPU=`ls /dev/accel/accel[0-9] | wc -l`
IDLE_HPU=$(($AVAILABLE_HPU - $hpu_num))
LOCK_FILE=/tmp/hpu.lock
touch ${LOCK_FILE}
for ((i=0; i<${IDLE_HPU}; i++)); do
        echo "Starting keep alive process num ${i}"
        python ./keep_alive.py ${LOCK_FILE} > /dev/null 2>/dev/null &
done

python -m vllm.entrypoints.openai.api_server --port 8084 \
        --model $model \
        --tensor-parallel-size $hpu_num \
        --max-num-seqs $batch_size \
        --disable-log-requests \
        --dtype bfloat16 \
        --block-size $block_size \
        --max-model-len $max_model_len \
        --gpu-memory-utilization 0.98 \
        --chat-template=/root/examples/vllm/template/mistral_mixtral.jinja \
        --swap-space 16 \
        $EAGER_FLAG \
        $DELAY_SAMPLING_FLAG \
        $QUANT_FLAGS >> ${output_dir}/vllm_server.log 2>&1 &

wait_for_server 8084 $model
if [[ $? -ne 0 ]]; then
    echo "Error: Server on port 8084 failed to start."
    exit 1
fi
