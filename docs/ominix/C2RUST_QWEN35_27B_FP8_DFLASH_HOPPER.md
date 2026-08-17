# C2Rust Qwen3.5-27B Block-FP8 + DFlash on Hopper

This runbook reproduces the validated single-session C2Rust runtime on one
NVIDIA GH200 and describes its intended headless boundary below OminiX-API. It
serves a serialized 128 x 128 block-FP8 target with a six-layer BF16 DFlash
draft. The recorded performance and model-output acceptance used SGLang's
direct loopback HTTP mode; the complete OminiX-API-to-gRPC chain remains an
integration gate for this model recipe.

The target is FP8 E4M3 with dynamic activations; it is not GGUF Q8. Modules in
the official Qwen3.5-27B exclusion list remain BF16, including the GDN inputs
that cannot use the tiled FP8 path. The converted checkpoint contained 400 FP8
weights, 400 FP32 inverse-scale tensors, and 259 excluded modules.

## Runtime pin

The OminiX-SGLang snapshot from which this recipe was prepared does not yet
define the `DFLASH` speculative algorithm. Do not run the recipe against that
snapshot's Python package and assume it is equivalent. The reproducible path
pins the tested SGLang v0.5.16 CUDA 12.9 image by digest:

```text
lmsysorg/sglang@sha256:b688781f3ef66522365ec570885a064b6734750ee22f0d32b3ed49aad87fbf90
```

The small derived image only installs `accelerate==1.14.0`, which Transformers
uses for the conversion-time `device_map`. The measured environment used
SGLang 0.5.16, PyTorch 2.11.0+cu129, Transformers 5.12.1, Accelerate 1.14.0,
and Safetensors 0.8.0.

Build it from the repository root:

```bash
docker build \
  --file docker/ominix-c2rust-hopper.Dockerfile \
  --tag ominix-c2rust-hopper:0.5.16 \
  .
```

The model checkpoints are not part of this repository. Before using the
commands below, set these variables to existing absolute host directories; the
output parent must be writable:

```bash
: "${C2RUST_BF16_CHECKPOINT:?set the source C2Rust BF16 checkpoint directory}"
: "${C2RUST_FP8_OUTPUT_PARENT:?set an existing parent for the converted checkpoint}"
: "${DFLASH_BF16_CHECKPOINT:?set the six-layer BF16 DFlash checkpoint directory}"

C2RUST_FP8_CHECKPOINT_NAME=${C2RUST_FP8_CHECKPOINT_NAME:-C2Rust-FP8-BLOCK128}
C2RUST_FP8_CHECKPOINT="$C2RUST_FP8_OUTPUT_PARENT/$C2RUST_FP8_CHECKPOINT_NAME"
test ! -e "$C2RUST_FP8_CHECKPOINT"
```

The recorded run used these immutable Apache-2.0 checkpoints:

| Role | Hugging Face repository | Revision | `config.json` SHA-256 |
| --- | --- | --- | --- |
| C2Rust BF16 source | `moxin-org/C2Rust` | `8d9d4cb3b8a24befbf636a2ad0d463db166a2dbb` | `16a5b35797dd7600799788536cec19419fbcd882efa4cfb86de1ed56a30a9f93` |
| Qwen3.5-27B DFlash BF16 draft | `z-lab/Qwen3.5-27B-DFlash` | `25ee0025ff950496a634e100b75c2db4515e9824` | `4ddb400eef4c5bb724d60410d04b84d17a84769c5bbef61391fbf6413e5139e3` |

Download those exact revisions with the Hugging Face CLI:

```bash
hf download moxin-org/C2Rust \
  --revision 8d9d4cb3b8a24befbf636a2ad0d463db166a2dbb \
  --local-dir "$C2RUST_BF16_CHECKPOINT"

hf download z-lab/Qwen3.5-27B-DFlash \
  --revision 25ee0025ff950496a634e100b75c2db4515e9824 \
  --local-dir "$DFLASH_BF16_CHECKPOINT"
```

The converter also pins the source safetensors index SHA-256 to
`54adabdb313601233d9cbdd43a5f6ed8594f94bb95ad50459bdfe23fae901428`
and its two shard hashes to
`5c8b98e5e90ea8e561af0e82ab678af3a0127d66d95904dc015640dcdf34391a`
and
`8421f39f527ae2d09cebdd8561c473670f80a1400d4680b723fd5ba03068925c`.
Use `--allow-unpinned-source` only for a deliberate variant that will receive
its own accuracy and performance evaluation. The launcher similarly checks the
validated target configuration SHA-256
`ff57958b36f92f1d60a847c097a1ec0248aecd2690b7a1aa7cc93cbda1625b10` and
the draft configuration hash shown above and draft weight SHA-256
`d8aa8e4043c0af29067a89c1f2cd8e01f9a7b70d6c794a070362bd8347071055`
by default; its explicit escape hatch is `C2RUST_ALLOW_UNPINNED_MODELS=1`.

## Convert and validate the target

The converter refuses an existing output or stale `.partial` directory. It
writes to a sibling staging directory, validates the architecture,
quantization metadata, FP8/scale pairing and shapes, BF16 GDN exclusions, and
tokenizer/processor asset hashes, then atomically publishes the checkpoint.

```bash
docker run --rm \
  --gpus all \
  --ipc host \
  --entrypoint python3 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$C2RUST_BF16_CHECKPOINT",dst=/models/source,readonly \
  --mount type=bind,src="$C2RUST_FP8_OUTPUT_PARENT",dst=/models/output-root \
  --workdir /workspace \
  ominix-c2rust-hopper:0.5.16 \
  scripts/ominix/c2rust_fp8_block128_convert.py \
  --source /models/source \
  --output "/models/output-root/$C2RUST_FP8_CHECKPOINT_NAME"
```

An existing converted checkpoint can be checked without loading it on the GPU:

```bash
docker run --rm \
  --entrypoint python3 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$C2RUST_BF16_CHECKPOINT",dst=/models/source,readonly \
  --mount type=bind,src="$C2RUST_FP8_CHECKPOINT",dst=/models/output,readonly \
  --workdir /workspace \
  ominix-c2rust-hopper:0.5.16 \
  scripts/ominix/c2rust_fp8_block128_convert.py \
  --source /models/source \
  --output /models/output \
  --validate-only
```

## Validated SGLang configuration

The launcher defaults to DFlash and the headless gRPC transport. Its validated
single-GPU runtime settings are:

- block-FP8 target and unquantized BF16 DFlash draft;
- FlashInfer attention and `flashinfer_deepgemm` target FP8 GEMM;
- Triton draft attention and DFlash block size 16;
- one running request and one decode CUDA-graph batch slot;
- 32,768-token context, 0.70 static-memory fraction, and language-only mode;
- Triton linear-attention decode, FlashInfer linear-attention prefill, and FP32
  Mamba state;
- thinking disabled in the default chat-template arguments.

Start the scheduler in a container on loopback:

```bash
docker run --rm \
  --name ominix-c2rust-scheduler \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --entrypoint bash \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env C2RUST_MODEL_PATH=/models/target \
  --env DFLASH_MODEL_PATH=/models/draft \
  --env C2RUST_SERVED_MODEL_NAME=C2Rust-FP8-DFlash \
  --env C2RUST_TRANSPORT=grpc \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$C2RUST_FP8_CHECKPOINT",dst=/models/target,readonly \
  --mount type=bind,src="$DFLASH_BF16_CHECKPOINT",dst=/models/draft,readonly \
  --workdir /workspace \
  ominix-c2rust-hopper:0.5.16 \
  scripts/ominix/launch_c2rust_fp8_dflash.sh dflash
```

The launcher binds to `127.0.0.1:30000` by default and refuses a non-loopback
host unless explicitly overridden.

## Intended OminiX serving boundary (integration gate)

The intended production ownership boundary is:

```text
client
  -> OminiX-API OpenAI endpoint
  -> authenticated OminiX worker-v0 HTTP/SSE shim
  -> SGLang scheduler gRPC on loopback
  -> C2Rust block-FP8 target + BF16 DFlash draft
```

SGLang remains headless; OminiX-API owns the public HTTP/OpenAI contract. The
full chain below is the intended integration target, not evidence for the
direct-runtime performance table. Run the shim in the same pinned image so its
`grpcio`, protobuf, `smg-grpc-proto`, Transformers, and tokenizer dependencies
match the scheduler. The token is read from the environment rather than exposed
in the process argument list:

```bash
: "${OMINIX_SCHEDULER_TOKEN:?set an internal scheduler token}"

docker run --rm \
  --name ominix-c2rust-v0-shim \
  --network host \
  --entrypoint python3 \
  --env OMINIX_SCHEDULER_TOKEN \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$C2RUST_FP8_CHECKPOINT",dst=/models/target,readonly \
  --workdir /workspace \
  ominix-c2rust-hopper:0.5.16 \
  scripts/ominix/v0_http_scheduler_shim.py \
  --mode grpc \
  --grpc-target 127.0.0.1:30000 \
  --host 127.0.0.1 \
  --port 19091 \
  --tokenizer-path /models/target \
  --model-id C2Rust-FP8-DFlash \
  --token-env OMINIX_SCHEDULER_TOKEN
```

OminiX-API runs on macOS/Apple Silicon. If the CUDA worker is remote, carry the
loopback shim listener over a managed SSH tunnel (or expose it through an
authenticated HTTPS service). Keep this tunnel running in its own terminal:

```bash
ssh -o ExitOnForwardFailure=yes -N -L 19091:127.0.0.1:19091 cuda-worker.example
```

Point the standard OminiX-API process at the local end of that tunnel:

```bash
: "${OMINIX_SCHEDULER_TOKEN:?set the same internal token on the API host}"
: "${OMINIX_API_BIN:=ominix-api}"

OMINIX_V0_SCHEDULER_URL=http://127.0.0.1:19091 \
OMINIX_V0_SCHEDULER_MODELS=C2Rust-FP8-DFlash \
OMINIX_V0_SERVED_MODEL=C2Rust-FP8-DFlash \
OMINIX_V0_SCHEDULER_TOKEN="$OMINIX_SCHEDULER_TOKEN" \
OMINIX_V0_SCHEDULER_TIMEOUT_SECS=1800 \
OMINIX_V0_CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}' \
OMINIX_API_HOST=127.0.0.1 \
PORT="${OMINIX_API_PORT:-8080}" \
"$OMINIX_API_BIN"
```

OminiX-API currently has no public-client authentication. Keep it on loopback,
or put an authenticated TLS reverse proxy in front of it before changing the
bind host. The shim token protects only API-to-shim traffic; it is not a public
API key. Exact model aliases are configured explicitly—CUDA is never selected
merely because a GPU is present—and a mapped worker failure never falls back to
the local MLX engine.

## Direct loopback benchmark

Direct HTTP is a diagnostic mode for measuring the SGLang runtime without the
OminiX boundary. Stop the gRPC scheduler first, then run the same container
command with `C2RUST_TRANSPORT=http`. Do not expose this endpoint publicly.

```bash
docker run --rm \
  --name ominix-c2rust-benchmark \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --entrypoint bash \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env C2RUST_MODEL_PATH=/models/target \
  --env DFLASH_MODEL_PATH=/models/draft \
  --env C2RUST_SERVED_MODEL_NAME=C2Rust-FP8-DFlash \
  --env C2RUST_TRANSPORT=http \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$C2RUST_FP8_CHECKPOINT",dst=/models/target,readonly \
  --mount type=bind,src="$DFLASH_BF16_CHECKPOINT",dst=/models/draft,readonly \
  --workdir /workspace \
  ominix-c2rust-hopper:0.5.16 \
  scripts/ominix/launch_c2rust_fp8_dflash.sh dflash
```

For the target-only reference, use the same command without the draft mount or
`DFLASH_MODEL_PATH`, set `C2RUST_SERVED_MODEL_NAME=C2Rust-FP8`, and finish the
command with `scripts/ominix/launch_c2rust_fp8_dflash.sh target`. Use
`C2Rust-FP8` as the benchmark model name.

The fixed-length performance run uses one 95-token C-to-Rust prompt, forces
512 generated tokens, temperature 0, seed 42, two warmups, and ten measured
requests. Requests are sequential, so the result is single-session throughput,
not batched aggregate throughput.

```bash
python3 scripts/ominix/benchmark_c2rust_fp8_dflash.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model C2Rust-FP8-DFlash \
  --cases find_max \
  --warmups 2 \
  --repeats 10 \
  --max-tokens 512 \
  --force-length \
  --sglang-meta \
  --output performance-results.json
```

For normal EOS behavior and the five-case semantic suite:

```bash
python3 scripts/ominix/benchmark_c2rust_fp8_dflash.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model C2Rust-FP8-DFlash \
  --cases all \
  --warmups 2 \
  --repeats 3 \
  --max-tokens 512 \
  --output results.json
```

Generated model output is untrusted code. Build the pinned, non-root validator
image and execute it with no network, a read-only root filesystem, no Linux
capabilities, bounded processes/memory/CPU, and only the result file mounted
read-only:

```bash
docker build \
  --file docker/ominix-c2rust-validator.Dockerfile \
  --tag ominix-c2rust-validator:1.97.1 \
  .

if [ "$(id -u)" -eq 0 ]; then
  echo "Refusing to execute model-generated code as root." >&2
  false
else
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 64 \
    --memory 1g \
    --cpus 1 \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=512m,mode=1777 \
    --mount type=bind,src="$PWD/results.json",dst=/input/results.json,readonly \
    ominix-c2rust-validator:1.97.1 \
    /input/results.json \
    > validation.json
fi
```

The validator refuses execution unless its image-supplied
`--execute-generated-code` acknowledgement is present. It also requires a
successful normal-EOS benchmark containing all five cases and the configured
repeat count; `--allow-partial-suite` is available only for explicit diagnostic
subsets. It strips optional Markdown fences, deduplicates outputs by hash,
compiles every unique answer with `rustc --test`, and executes its case-specific
semantic tests. A target-only normal-EOS result can also be supplied with
`--reference-json` to report byte-level matches without turning them into the
semantic pass criterion.
When using that option in the container, bind-mount the reference JSON
read-only under `/input` as well and pass its container path.

## Measured result

The recorded direct-runtime measurements used one NVIDIA GH200 with 97,871 MiB
HBM and compute capability 9.0. Wall throughput includes the complete
client-observed request; decode throughput is SGLang's internal decode metric.

| Configuration | Wall output throughput | Internal decode | Relative result |
| --- | ---: | ---: | ---: |
| llama.cpp GGUF Q8_0 + DFlash baseline | 306.753 tok/s | not recorded | baseline |
| SGLang block-FP8 target only | 84.626 tok/s | not recorded | target-only reference |
| SGLang block-FP8 + BF16 DFlash, validated settings | 395.976 tok/s | 487.463 tok/s | +29.1% vs llama.cpp; 4.68x target-only |

The five normal-EOS cases produced 15 measured Rust answers. Every unique answer
representing those runs compiled and passed its semantic tests. A separate
request with a 30,015-token prompt also succeeded with the 32,768-token server
context. Normal-EOS throughput is not directly comparable to the fixed
512-token table because answer lengths differ.

Target-only and speculative decoding need not be byte-identical at every
numerically close decision. In the acceptance suite, one `checked_sum` answer
differed textually while both variants compiled and passed the same semantic
tests. Semantic execution is therefore the correctness gate; output hashes are
reported for reproducibility and diagnostics.

## Required workarounds

- `--mamba-radix-cache-strategy no_buffer` must be paired with
  `--disable-overlap-schedule` for this path.
- Set `--speculative-draft-model-quantization unquant` explicitly. Otherwise
  the target's FP8 quantization can propagate to the draft. The online-FP8
  draft was also slightly slower than the BF16 draft in the matched benchmark.
- Do not enable GDN ReplaySSM speculative state. SGLang v0.5.16's
  `DFlashWorkerV2` bypasses the generic ReplaySSM commit and can scatter a
  missing intermediate state. The launcher rejects this combination.
- `--disable-prefill-cuda-graph` reduces cold-start time and HBM use. It does
  not disable the target verification or draft decode graphs used by DFlash.
- FlashInfer draft attention was slower in the matched test, so the validated
  configuration keeps Triton for draft attention.

## Rejected TileLang/TileIR experiment

A bit-exact Hopper Q8 TMA/WGMMA prototype was generated from a TileLang/TileIR
experiment and evaluated in a separate Q8 kernel harness; those figures are not
comparable to the end-to-end generation table above. The stock path measured
619.24-620.16 tok/s, while the prototype measured 613.34-614.34 tok/s, about 1%
slower. It also consumed an additional 11,776 MiB of HBM. The prototype was
therefore not shipped, and its experimental patch is intentionally excluded
from this repository.

## Scope

This recipe intentionally excludes model weights, host addresses, service
manager units, credentials, raw benchmark JSON, and local deployment paths.
Persisted benchmark artifacts may contain generated code and runtime metadata;
review them before sharing.
