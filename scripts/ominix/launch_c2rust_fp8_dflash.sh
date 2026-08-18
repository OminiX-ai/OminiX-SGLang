#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: launch_c2rust_fp8_dflash.sh [--dry-run|--check] [target|dflash]

Required environment:
  C2RUST_MODEL_PATH       Serialized Qwen3.5/C2Rust block-FP8 checkpoint
  DFLASH_MODEL_PATH       BF16 DFlash checkpoint (dflash mode only)

The default transport is headless gRPC. Set C2RUST_TRANSPORT=http only for a
loopback benchmark endpoint; OminiX-API should own the public OpenAI API.
EOF
}

dry_run=0
check_only=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
elif [[ "${1:-}" == "--check" ]]; then
    check_only=1
    shift
fi
mode=${1:-dflash}
if (($# > 0)); then
    shift
fi
if (($# > 0)); then
    usage >&2
    exit 2
fi
if [[ "$mode" != "target" && "$mode" != "dflash" ]]; then
    echo "unsupported mode: $mode" >&2
    usage >&2
    exit 2
fi

: "${C2RUST_MODEL_PATH:?C2RUST_MODEL_PATH is required}"
if [[ "$mode" == "dflash" ]]; then
    : "${DFLASH_MODEL_PATH:?DFLASH_MODEL_PATH is required in dflash mode}"
fi

python_bin=${C2RUST_PYTHON_BIN:-python3}
transport=${C2RUST_TRANSPORT:-grpc}
host=${C2RUST_HOST:-127.0.0.1}
port=${C2RUST_PORT:-30000}
if [[ "$mode" == "dflash" ]]; then
    default_model_name=C2Rust-FP8-DFlash
else
    default_model_name=C2Rust-FP8
fi
served_model_name=${C2RUST_SERVED_MODEL_NAME:-$default_model_name}
context_length=${C2RUST_CONTEXT_LENGTH:-262144}
mem_fraction_static=${C2RUST_MEM_FRACTION_STATIC:-0.70}
chunked_prefill_size=${C2RUST_CHUNKED_PREFILL_SIZE:-8192}
watchdog_timeout=${C2RUST_WATCHDOG_TIMEOUT:-1800}
tp_size=${C2RUST_TP_SIZE:-1}
disable_prefill_graph=${C2RUST_DISABLE_PREFILL_CUDA_GRAPH:-1}
dflash_block_size=${C2RUST_DFLASH_BLOCK_SIZE:-16}

if [[ "$transport" != "grpc" && "$transport" != "http" ]]; then
    echo "C2RUST_TRANSPORT must be grpc or http, got: $transport" >&2
    exit 2
fi
if [[ "$host" != "127.0.0.1" && "$host" != "localhost" && "${C2RUST_ALLOW_NON_LOOPBACK:-0}" != "1" ]]; then
    echo "refusing non-loopback host without C2RUST_ALLOW_NON_LOOPBACK=1" >&2
    exit 2
fi
for pair in \
    "C2RUST_PORT:$port" \
    "C2RUST_CONTEXT_LENGTH:$context_length" \
    "C2RUST_CHUNKED_PREFILL_SIZE:$chunked_prefill_size" \
    "C2RUST_WATCHDOG_TIMEOUT:$watchdog_timeout" \
    "C2RUST_TP_SIZE:$tp_size" \
    "C2RUST_DFLASH_BLOCK_SIZE:$dflash_block_size"; do
    name=${pair%%:*}
    value=${pair#*:}
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got: $value" >&2
        exit 2
    fi
done
if ((10#$port > 65535)); then
    echo "C2RUST_PORT must be at most 65535, got: $port" >&2
    exit 2
fi
if [[ ! "$mem_fraction_static" =~ ^0\.[0-9]*[1-9][0-9]*$ ]]; then
    echo "C2RUST_MEM_FRACTION_STATIC must be greater than 0 and less than 1" >&2
    exit 2
fi
if [[ "$disable_prefill_graph" != "0" && "$disable_prefill_graph" != "1" ]]; then
    echo "C2RUST_DISABLE_PREFILL_CUDA_GRAPH must be 0 or 1" >&2
    exit 2
fi
if [[ "${C2RUST_ENABLE_GDN_REPLAYSSM_SPEC:-0}" == "1" ]]; then
    echo "GDN ReplaySSM is not compatible with DFlashWorkerV2 in SGLang v0.5.16" >&2
    exit 2
fi

args=(
    "$python_bin" -m sglang.launch_server
    --model-path "$C2RUST_MODEL_PATH"
    --served-model-name "$served_model_name"
    --host "$host"
    --port "$port"
    --quantization fp8
    --dtype bfloat16
    --tp-size "$tp_size"
    --attention-backend flashinfer
    --fp8-gemm-backend flashinfer_deepgemm
    --max-running-requests 1
    --cuda-graph-max-bs-decode 1
    --linear-attn-decode-backend triton
    --linear-attn-prefill-backend flashinfer
    --mamba-radix-cache-strategy no_buffer
    --disable-overlap-schedule
    --mamba-ssm-dtype float32
    --language-only
    --context-length "$context_length"
    --mem-fraction-static "$mem_fraction_static"
    --chunked-prefill-size "$chunked_prefill_size"
    --default-chat-template-kwargs '{"enable_thinking": false}'
    --enable-metrics
    --log-level info
    --watchdog-timeout "$watchdog_timeout"
)

if [[ "$transport" == "grpc" ]]; then
    args+=(--smg-grpc-mode --grpc-port "$port")
fi
if [[ "$disable_prefill_graph" == "1" ]]; then
    args+=(--disable-prefill-cuda-graph)
fi
if [[ "$mode" == "dflash" ]]; then
    args+=(
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "$DFLASH_MODEL_PATH"
        --speculative-draft-model-quantization unquant
        --speculative-dflash-block-size "$dflash_block_size"
        --speculative-draft-attention-backend triton
    )
fi

if (( ! dry_run )) && [[ "${C2RUST_ALLOW_UNPINNED_MODELS:-0}" != "1" ]]; then
    target_config_sha=ff57958b36f92f1d60a847c097a1ec0248aecd2690b7a1aa7cc93cbda1625b10
    draft_config_sha=4ddb400eef4c5bb724d60410d04b84d17a84769c5bbef61391fbf6413e5139e3
    draft_weights_sha=d8aa8e4043c0af29067a89c1f2cd8e01f9a7b70d6c794a070362bd8347071055
    file_sha256() {
        "$python_bin" -c \
            'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); [h.update(b) for b in iter(lambda:f.read(8<<20), b"")]; print(h.hexdigest())' \
            "$1"
    }
    actual_target_sha=$(file_sha256 "$C2RUST_MODEL_PATH/config.json")
    if [[ "$actual_target_sha" != "$target_config_sha" ]]; then
        echo "target config does not match the validated C2Rust block-FP8 artifact" >&2
        exit 2
    fi
    if [[ "$mode" == "dflash" ]]; then
        actual_draft_sha=$(file_sha256 "$DFLASH_MODEL_PATH/config.json")
        if [[ "$actual_draft_sha" != "$draft_config_sha" ]]; then
            echo "draft config does not match the validated Qwen3.5-27B DFlash artifact" >&2
            exit 2
        fi
        actual_draft_weights_sha=$(file_sha256 "$DFLASH_MODEL_PATH/model.safetensors")
        if [[ "$actual_draft_weights_sha" != "$draft_weights_sha" ]]; then
            echo "draft weights do not match the validated Qwen3.5-27B DFlash artifact" >&2
            exit 2
        fi
    fi
fi

if ((dry_run || check_only)); then
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
fi

exec "${args[@]}"
