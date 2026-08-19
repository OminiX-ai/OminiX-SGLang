#!/bin/bash
# Qwen3.8 parity launcher: same optimization stack as C2Rust prod
# (thinking OFF, NGRAM draft 32 / sam 31, max-running-requests 4, corpus optional).
set -e
sudo docker rm -f qwen-ab >/dev/null 2>&1 || true
MRR="${ABMRR:-4}"
DRAFT="${ABDRAFT:-32}"
SAM=$((DRAFT - 1))
CORPUS_ARGS=()
if [ -f /home/ubuntu/qwen38-h200/ngram-corpus/cards.jsonl ]; then
  CORPUS_ARGS=(-v /home/ubuntu/qwen38-h200/ngram-corpus:/corpus:ro)
  SPEC_CORPUS=(--speculative-ngram-external-corpus-path /corpus/cards.jsonl --speculative-ngram-external-sam-budget "$SAM")
else
  SPEC_CORPUS=()
fi
sudo docker run -d --name qwen-ab --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --cap-add SYS_NICE \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  -v /home/ubuntu/qwen38-h200/models/Qwen3.8-27B-FP8-017b9c7:/models/target:ro \
  "${CORPUS_ARGS[@]}" \
  ominix-sglang-hopper:0.5.16-qwen38 \
  python3 -m sglang.launch_server \
    --model-path /models/target --served-model-name Qwen3.8-27B-FP8-DFlash \
    --host 0.0.0.0 --port 30878 --trust-remote-code --load-format safetensors \
    --quantization fp8 --dtype bfloat16 --tp-size 1 \
    --attention-backend flashinfer --fp8-gemm-backend flashinfer_deepgemm \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests "$MRR" --cuda-graph-max-bs-decode "$MRR" \
    --disable-prefill-cuda-graph \
    --linear-attn-decode-backend triton --linear-attn-prefill-backend flashinfer \
    --mamba-radix-cache-strategy "${ABSTRAT:-extra_buffer}" $( [ "${ABOVERLAP:-1}" = "0" ] && printf -- "--disable-overlap-schedule" ) --mamba-ssm-dtype float32 \
    --language-only --context-length 262144 --mem-fraction-static "${ABMEM:-0.75}" \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
$( [ "${ABPARSERS:-1}" = "1" ] && printf -- "--reasoning-parser qwen3 --tool-call-parser qwen3_coder" ) \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-metrics --log-level info --watchdog-timeout 1800 \
$( [ -n "${ABMAMBA:-}" ] && printf -- "--max-mamba-cache-size %s" "$ABMAMBA" ) \
    --speculative-algorithm NGRAM --speculative-num-draft-tokens "$DRAFT" \
    "${SPEC_CORPUS[@]}"
echo "launched qwen-ab: mrr=$MRR draft=$DRAFT corpus=$([ -f /home/ubuntu/qwen38-h200/ngram-corpus/cards.jsonl ] && echo yes || echo no)"
