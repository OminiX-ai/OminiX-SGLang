#!/usr/bin/env python3
"""Convert C2Rust/Qwen3.5-27B to serialized 128x128 block-FP8.

The conversion follows Qwen/Qwen3.5-27B-FP8's exclusion policy and validates
the complete serialized checkpoint before returning success.  Heavy runtime
imports are intentionally delayed so ``--help`` and unit tests remain usable
outside the conversion container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

ASSET_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_EXCLUSION_COUNT = 259
EXPECTED_TEXT_LAYERS = 64
EXPECTED_LINEAR_ATTENTION_LAYERS = 48
EXPECTED_VISION_LAYERS = 27
EXPECTED_FP8_WEIGHT_COUNT = 400
EXPECTED_SOURCE_REVISION = "8d9d4cb3b8a24befbf636a2ad0d463db166a2dbb"
EXPECTED_SOURCE_FILES_SHA256 = {
    "config.json": "16a5b35797dd7600799788536cec19419fbcd882efa4cfb86de1ed56a30a9f93",
    "model.safetensors.index.json": (
        "54adabdb313601233d9cbdd43a5f6ed8594f94bb95ad50459bdfe23fae901428"
    ),
    "model-00001-of-00002.safetensors": (
        "5c8b98e5e90ea8e561af0e82ab678af3a0127d66d95904dc015640dcdf34391a"
    ),
    "model-00002-of-00002.safetensors": (
        "8421f39f527ae2d09cebdd8561c473670f80a1400d4680b723fd5ba03068925c"
    ),
}


def validate_source_config(config: Any) -> list[str]:
    """Validate the exact C2Rust/Qwen3.5-27B architecture contract."""
    if config.architectures != [EXPECTED_ARCHITECTURE]:
        raise ValueError(f"Unexpected source architecture: {config.architectures}")
    return official_qwen35_27b_exclusions(config)


def official_qwen35_27b_exclusions(config: Any) -> list[str]:
    """Build the official Qwen3.5-27B modules-to-not-convert list."""
    text_config = config.get_text_config()
    layer_types = list(text_config.layer_types)
    linear_attention_layers = layer_types.count("linear_attention")
    if len(layer_types) != EXPECTED_TEXT_LAYERS:
        raise ValueError(
            f"Expected {EXPECTED_TEXT_LAYERS} text layers, got {len(layer_types)}"
        )
    if linear_attention_layers != EXPECTED_LINEAR_ATTENTION_LAYERS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_LINEAR_ATTENTION_LAYERS} linear-attention layers, "
            f"got {linear_attention_layers}"
        )
    if config.vision_config.depth != EXPECTED_VISION_LAYERS:
        raise ValueError(
            f"Expected vision depth {EXPECTED_VISION_LAYERS}, "
            f"got {config.vision_config.depth}"
        )
    excluded = ["lm_head", "model.language_model.embed_tokens"]

    for layer_id, layer_type in enumerate(text_config.layer_types):
        if layer_type != "linear_attention":
            continue
        prefix = f"model.language_model.layers.{layer_id}.linear_attn"
        excluded.extend(
            (
                f"{prefix}.conv1d",
                f"{prefix}.in_proj_a",
                f"{prefix}.in_proj_b",
            )
        )

    for layer_id in range(config.vision_config.depth):
        prefix = f"model.visual.blocks.{layer_id}"
        excluded.extend(
            (
                f"{prefix}.attn.proj",
                f"{prefix}.attn.qkv",
                f"{prefix}.mlp.linear_fc1",
                f"{prefix}.mlp.linear_fc2",
            )
        )

    excluded.extend(
        (
            "model.visual.merger.linear_fc1",
            "model.visual.merger.linear_fc2",
            "model.visual.patch_embed.proj",
            "model.visual.pos_embed",
            "mtp.fc",
        )
    )
    result = sorted(set(excluded))
    if len(result) != EXPECTED_EXCLUSION_COUNT:
        raise ValueError(
            "Expected the Qwen3.5-27B official "
            f"{EXPECTED_EXCLUSION_COUNT} exclusions, got {len(result)}"
        )
    return result


def prepare_partial_output(output: Path) -> Path:
    """Create a sibling staging directory for an atomic checkpoint publish."""
    partial = output.with_name(f"{output.name}.partial")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if partial.exists():
        raise FileExistsError(f"Stale partial output exists: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


def copy_assets(source: Path, output: Path) -> None:
    for name in ASSET_FILES:
        source_path = source / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Required source asset is missing: {source_path}")
        shutil.copy2(source_path, output / name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_identity(source: Path, allow_unpinned: bool) -> None:
    """Require the exact source revision used for the recorded acceptance."""
    if allow_unpinned:
        return
    for name, expected_hash in EXPECTED_SOURCE_FILES_SHA256.items():
        path = source / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(
                f"Source {name} does not match moxin-org/C2Rust revision "
                f"{EXPECTED_SOURCE_REVISION}; pass --allow-unpinned-source "
                "only for an intentional, separately evaluated variant"
            )


def checkpoint_shards(checkpoint: Path) -> list[Path]:
    """Return checkpoint shards after verifying the index-to-file contract."""
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors weight_map: {index_path}")
        shard_names: set[str] = set()
        for key, name in weight_map.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"Invalid tensor key in {index_path}")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"Invalid shard name in {index_path}: {name!r}")
            shard_names.add(name)
        shards = [checkpoint / name for name in sorted(shard_names)]
        missing = [path.name for path in shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint shard(s): {missing}")
        actual_names = {path.name for path in checkpoint.glob("*.safetensors")}
        if actual_names != shard_names:
            raise ValueError(
                "Safetensors files differ from the shard index: "
                f"indexed={sorted(shard_names)}, actual={sorted(actual_names)}"
            )
        return shards
    single = checkpoint / "model.safetensors"
    if single.is_file():
        actual_names = {path.name for path in checkpoint.glob("*.safetensors")}
        if actual_names != {single.name}:
            raise ValueError(f"Unexpected safetensors files: {sorted(actual_names)}")
        return [single]
    raise FileNotFoundError(f"No safetensors checkpoint/index found in {checkpoint}")


def tensor_metadata(checkpoint: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read tensor metadata and prove that every index entry maps correctly."""
    from safetensors import safe_open

    metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    locations: dict[str, str] = {}
    for shard in checkpoint_shards(checkpoint):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in metadata:
                    raise ValueError(f"Duplicate tensor key: {key}")
                view = handle.get_slice(key)
                metadata[key] = (str(view.get_dtype()), tuple(view.get_shape()))
                locations[key] = shard.name

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if set(weight_map) != set(metadata):
            missing = sorted(set(weight_map) - set(metadata))
            unindexed = sorted(set(metadata) - set(weight_map))
            raise ValueError(
                "Safetensors index keys differ from shard contents: "
                f"missing={missing[:5]}, unindexed={unindexed[:5]}"
            )
        misplaced = [
            key
            for key, shard_name in weight_map.items()
            if locations[key] != shard_name
        ]
        if misplaced:
            raise ValueError(f"Tensor(s) stored in the wrong shard: {misplaced[:5]}")
    return metadata


def is_fp8_e4m3(dtype: str) -> bool:
    upper = dtype.upper()
    return "F8_E4M3" in upper or "FLOAT8_E4M3" in upper


def is_fp32(dtype: str) -> bool:
    return dtype.upper() in {"F32", "FLOAT32", "TORCH.FLOAT32"}


def is_bf16(dtype: str) -> bool:
    return dtype.upper() in {"BF16", "BFLOAT16", "TORCH.BFLOAT16"}


def canonical_dtype(dtype: str) -> str:
    upper = dtype.upper()
    aliases = {
        "BFLOAT16": "BF16",
        "TORCH.BFLOAT16": "BF16",
        "FLOAT32": "F32",
        "TORCH.FLOAT32": "F32",
    }
    return aliases.get(upper, upper)


def is_excluded_tensor(key: str, exclusions: list[str]) -> bool:
    return any(key == module or key.startswith(f"{module}.") for module in exclusions)


def validate_tensor_transform(
    source_tensors: dict[str, tuple[str, tuple[int, ...]]],
    output_tensors: dict[str, tuple[str, tuple[int, ...]]],
    exclusions: list[str],
) -> None:
    """Prove that the output is the complete expected transform of the source."""
    scale_suffix = ".weight_scale_inv"
    output_base_keys = {key for key in output_tensors if not key.endswith(scale_suffix)}
    if output_base_keys != set(source_tensors):
        missing = sorted(set(source_tensors) - output_base_keys)
        unexpected = sorted(output_base_keys - set(source_tensors))
        raise ValueError(
            "Converted checkpoint tensor set differs from source: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    expected_fp8: set[str] = set()
    for key, (source_dtype, source_shape) in source_tensors.items():
        output_dtype, output_shape = output_tensors[key]
        if output_shape != source_shape:
            raise ValueError(
                f"Converted tensor shape changed: {key}: "
                f"source={source_shape}, output={output_shape}"
            )
        eligible = (
            key.endswith(".weight")
            and len(source_shape) == 2
            and all(dimension % 128 == 0 for dimension in source_shape)
            and not is_excluded_tensor(key, exclusions)
        )
        if eligible:
            expected_fp8.add(key)
            if not is_fp8_e4m3(output_dtype):
                raise ValueError(f"Eligible weight was not serialized as FP8: {key}")
        elif canonical_dtype(output_dtype) != canonical_dtype(source_dtype):
            raise ValueError(
                f"Retained tensor dtype changed: {key}: "
                f"source={source_dtype}, output={output_dtype}"
            )

    actual_fp8 = {
        key
        for key, (dtype, _) in output_tensors.items()
        if key.endswith(".weight") and is_fp8_e4m3(dtype)
    }
    if actual_fp8 != expected_fp8:
        raise ValueError(
            "FP8 tensor set differs from the expected block-128 transform: "
            f"missing={sorted(expected_fp8 - actual_fp8)[:5]}, "
            f"unexpected={sorted(actual_fp8 - expected_fp8)[:5]}"
        )
    expected_scales = {
        key.removesuffix(".weight") + scale_suffix for key in expected_fp8
    }
    actual_scales = {key for key in output_tensors if key.endswith(scale_suffix)}
    if actual_scales != expected_scales:
        raise ValueError(
            "FP8 inverse-scale tensor set differs from expectation: "
            f"missing={sorted(expected_scales - actual_scales)[:5]}, "
            f"unexpected={sorted(actual_scales - expected_scales)[:5]}"
        )


def validate_tensor_quantization(
    tensors: dict[str, tuple[str, tuple[int, ...]]], layer_types: list[str]
) -> tuple[int, int]:
    """Validate serialized FP8 pairs and critical retained-BF16 projections."""
    fp8_weights = 0
    scale_tensors = 0
    for key, (dtype, shape) in tensors.items():
        if key.endswith(".weight") and is_fp8_e4m3(dtype):
            fp8_weights += 1
            if len(shape) != 2:
                raise ValueError(f"FP8 weight is not 2-D: {key} {shape}")
            scale_key = key.removesuffix(".weight") + ".weight_scale_inv"
            if scale_key not in tensors:
                raise ValueError(f"FP8 weight has no inverse scale: {key}")
            scale_dtype, scale_shape = tensors[scale_key]
            expected_shape = (math.ceil(shape[0] / 128), math.ceil(shape[1] / 128))
            if not is_fp32(scale_dtype) or scale_shape != expected_shape:
                raise ValueError(
                    f"Bad scale for {key}: {scale_dtype} {scale_shape}; "
                    f"expected FP32 {expected_shape}"
                )
        elif key.endswith(".weight_scale_inv"):
            scale_tensors += 1
            weight_key = key.removesuffix(".weight_scale_inv") + ".weight"
            if weight_key not in tensors or not is_fp8_e4m3(tensors[weight_key][0]):
                raise ValueError(f"Orphan scale tensor: {key}")

    if fp8_weights != EXPECTED_FP8_WEIGHT_COUNT or fp8_weights != scale_tensors:
        raise ValueError(
            "Invalid FP8/scale counts: "
            f"expected={EXPECTED_FP8_WEIGHT_COUNT}, "
            f"weights={fp8_weights}, scales={scale_tensors}"
        )

    for layer_id, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        for module in ("in_proj_a", "in_proj_b"):
            key = f"model.language_model.layers.{layer_id}.linear_attn.{module}.weight"
            if key not in tensors or not is_bf16(tensors[key][0]):
                raise ValueError(f"Critical GDN exclusion is not BF16: {key}")
            if key.removesuffix(".weight") + ".weight_scale_inv" in tensors:
                raise ValueError(f"Critical GDN exclusion has an FP8 scale: {key}")

    return fp8_weights, scale_tensors


def validate(source: Path, output: Path, exclusions: list[str]) -> dict[str, Any]:
    from transformers import AutoConfig

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") or {}
    expected_fields = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "weight_per_tensor": False,
        "act_per_tensor": False,
        "scale_fmt": "float",
    }
    for key, expected in expected_fields.items():
        if quantization.get(key) != expected:
            raise ValueError(
                f"quantization_config.{key}: expected {expected!r}, "
                f"got {quantization.get(key)!r}"
            )
    if sorted(quantization.get("modules_to_not_convert", [])) != exclusions:
        raise ValueError("Serialized modules_to_not_convert differs from official list")
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise ValueError(f"Unexpected architectures: {config.get('architectures')}")

    source_tensors = tensor_metadata(source)
    tensors = tensor_metadata(output)
    validate_tensor_transform(source_tensors, tensors, exclusions)
    text_config = AutoConfig.from_pretrained(
        source, local_files_only=True
    ).get_text_config()
    fp8_weights, scale_tensors = validate_tensor_quantization(
        tensors, list(text_config.layer_types)
    )

    for name in ASSET_FILES:
        if sha256(source / name) != sha256(output / name):
            raise ValueError(f"Processor/tokenizer asset changed: {name}")

    total_bytes = sum(path.stat().st_size for path in checkpoint_shards(output))
    return {
        "status": "validated",
        "fp8_weights": fp8_weights,
        "fp32_scale_tensors": scale_tensors,
        "tensor_count": len(tensors),
        "checkpoint_bytes": total_bytes,
        "checkpoint_gib": round(total_bytes / (1 << 30), 3),
        "official_exclusion_count": len(exclusions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip conversion and validate an existing checkpoint.",
    )
    parser.add_argument(
        "--allow-unpinned-source",
        action="store_true",
        help="accept a same-shape source other than the recorded C2Rust revision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoConfig

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_source_identity(source, args.allow_unpinned_source)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    exclusions = validate_source_config(config)

    if not args.validate_only:
        import torch
        from transformers import AutoModelForImageTextToText, FineGrainedFP8Config

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for offline FP8 conversion")
        capability = torch.cuda.get_device_capability(0)
        if capability < (9, 0):
            raise RuntimeError(f"Hopper or newer is required; detected CC {capability}")
        partial = prepare_partial_output(output)

        quantization = FineGrainedFP8Config(
            activation_scheme="dynamic",
            weight_block_size=(128, 128),
            modules_to_not_convert=exclusions,
            scale_fmt="float",
        )
        quantization.weight_per_tensor = False
        quantization.act_per_tensor = False
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.eval()
        model.save_pretrained(partial, safe_serialization=True, max_shard_size="5GB")
        copy_assets(source, partial)
        del model
        torch.cuda.empty_cache()

        result = validate(source, partial, exclusions)
        partial.replace(output)
    else:
        result = validate(source, output, exclusions)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
