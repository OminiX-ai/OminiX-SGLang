#!/usr/bin/env python3
"""Dependency-light tests for the C2Rust FP8+DFlash recipe."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_c2rust_fp8_dflash as benchmark
import c2rust_fp8_block128_convert as converter
import validate_c2rust_rust_outputs as validator

LAUNCHER = SCRIPT_DIR / "launch_c2rust_fp8_dflash.sh"


class FakeTextConfig:
    layer_types = ["linear_attention"] * 48 + ["full_attention"] * 16


class FakeVisionConfig:
    depth = 27


class FakeConfig:
    architectures = [converter.EXPECTED_ARCHITECTURE]
    vision_config = FakeVisionConfig()

    @staticmethod
    def get_text_config() -> FakeTextConfig:
        return FakeTextConfig()


def valid_tensor_metadata() -> dict[str, tuple[str, tuple[int, ...]]]:
    metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    for index in range(converter.EXPECTED_FP8_WEIGHT_COUNT):
        prefix = f"quantized.layer_{index}"
        metadata[f"{prefix}.weight"] = ("F8_E4M3", (256, 384))
        metadata[f"{prefix}.weight_scale_inv"] = ("F32", (2, 3))
    for layer_id in range(48):
        for module in ("in_proj_a", "in_proj_b"):
            key = (
                f"model.language_model.layers.{layer_id}.linear_attn."
                f"{module}.weight"
            )
            metadata[key] = ("BF16", (48, 5120))
    return metadata


def valid_source_metadata() -> dict[str, tuple[str, tuple[int, ...]]]:
    metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    for index in range(converter.EXPECTED_FP8_WEIGHT_COUNT):
        metadata[f"quantized.layer_{index}.weight"] = ("BF16", (256, 384))
    for layer_id in range(48):
        for module in ("in_proj_a", "in_proj_b"):
            key = (
                f"model.language_model.layers.{layer_id}.linear_attn."
                f"{module}.weight"
            )
            metadata[key] = ("BF16", (48, 5120))
    return metadata


class ConverterUnitTest(unittest.TestCase):
    def test_help_does_not_import_conversion_dependencies(self) -> None:
        script = SCRIPT_DIR / "c2rust_fp8_block128_convert.py"
        result = subprocess.run(
            [sys.executable, "-S", str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--validate-only", result.stdout)

    def test_official_exclusions_match_qwen35_27b_contract(self) -> None:
        exclusions = converter.official_qwen35_27b_exclusions(FakeConfig())
        self.assertEqual(len(exclusions), 259)
        self.assertIn("model.language_model.layers.0.linear_attn.in_proj_a", exclusions)
        self.assertIn("model.visual.blocks.26.mlp.linear_fc2", exclusions)
        self.assertIn("mtp.fc", exclusions)

    def test_wrong_model_shape_is_rejected(self) -> None:
        config = FakeConfig()
        config.vision_config = type("Vision", (), {"depth": 26})()
        with self.assertRaisesRegex(ValueError, "vision depth"):
            converter.official_qwen35_27b_exclusions(config)

    def test_wrong_architecture_is_rejected(self) -> None:
        config = FakeConfig()
        config.architectures = ["DifferentArchitecture"]
        with self.assertRaisesRegex(ValueError, "Unexpected source architecture"):
            converter.validate_source_config(config)

    def test_serialized_fp8_tensor_contract(self) -> None:
        weights, scales = converter.validate_tensor_quantization(
            valid_tensor_metadata(), list(FakeTextConfig.layer_types)
        )
        self.assertEqual(weights, 400)
        self.assertEqual(scales, 400)
        converter.validate_tensor_transform(
            valid_source_metadata(),
            valid_tensor_metadata(),
            converter.official_qwen35_27b_exclusions(FakeConfig()),
        )

    def test_incomplete_converted_tensor_set_is_rejected(self) -> None:
        output = valid_tensor_metadata()
        del output["quantized.layer_0.weight"]
        del output["quantized.layer_0.weight_scale_inv"]
        with self.assertRaisesRegex(ValueError, "tensor set differs"):
            converter.validate_tensor_transform(
                valid_source_metadata(),
                output,
                converter.official_qwen35_27b_exclusions(FakeConfig()),
            )

    def test_bad_scale_shape_is_rejected(self) -> None:
        metadata = valid_tensor_metadata()
        metadata["quantized.layer_0.weight_scale_inv"] = ("F32", (2, 2))
        with self.assertRaisesRegex(ValueError, "Bad scale"):
            converter.validate_tensor_quantization(
                metadata, list(FakeTextConfig.layer_types)
            )

    def test_gdn_projection_must_remain_bf16(self) -> None:
        metadata = valid_tensor_metadata()
        key = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
        metadata[key] = ("F32", (48, 5120))
        with self.assertRaisesRegex(ValueError, "Critical GDN exclusion"):
            converter.validate_tensor_quantization(
                metadata, list(FakeTextConfig.layer_types)
            )

    def test_conversion_uses_fresh_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            output = Path(raw_temp) / "model"
            partial = converter.prepare_partial_output(output)
            self.assertEqual(partial.name, "model.partial")
            self.assertTrue(partial.is_dir())
            with self.assertRaisesRegex(FileExistsError, "Stale partial"):
                converter.prepare_partial_output(output)


class LauncherUnitTest(unittest.TestCase):
    def run_launcher(
        self, mode: str, *, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("C2RUST_") or key == "DFLASH_MODEL_PATH":
                env.pop(key)
        env.update(
            {
                "C2RUST_MODEL_PATH": "/models/target",
                "DFLASH_MODEL_PATH": "/models/draft",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run", mode],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_launcher_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_dflash_dry_run_contains_measured_winner(self) -> None:
        result = self.run_launcher("dflash")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout
        for expected in (
            "--smg-grpc-mode",
            "--grpc-port 30000",
            "--fp8-gemm-backend flashinfer_deepgemm",
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-quantization unquant",
            "--speculative-dflash-block-size 16",
            "--speculative-draft-attention-backend triton",
            "--mamba-radix-cache-strategy no_buffer",
            "--disable-overlap-schedule",
            "--disable-prefill-cuda-graph",
        ):
            self.assertIn(expected, command)
        self.assertNotIn("enable-gdn-replayssm-spec", command)
        self.assertNotIn("USE_TRITON_W8A8_FP8_KERNEL", command)

    def test_target_mode_omits_speculative_flags(self) -> None:
        result = self.run_launcher("target")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--speculative-", result.stdout)
        self.assertIn("--served-model-name C2Rust-FP8", result.stdout)

    def test_non_loopback_bind_requires_explicit_opt_in(self) -> None:
        result = self.run_launcher("dflash", extra_env={"C2RUST_HOST": "0.0.0.0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing non-loopback", result.stderr)

    def test_replayssm_combination_is_rejected(self) -> None:
        result = self.run_launcher(
            "dflash", extra_env={"C2RUST_ENABLE_GDN_REPLAYSSM_SPEC": "1"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not compatible", result.stderr)

    def test_dflash_requires_draft_checkpoint(self) -> None:
        env = os.environ.copy()
        env["C2RUST_MODEL_PATH"] = "/models/target"
        env.pop("DFLASH_MODEL_PATH", None)
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run", "dflash"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DFLASH_MODEL_PATH", result.stderr)


class BenchmarkUnitTest(unittest.TestCase):
    def test_endpoint_normalization_and_credential_rejection(self) -> None:
        self.assertEqual(
            benchmark.endpoint_from_base("http://localhost:30000/v1"),
            ("http", "localhost", 30000, "/v1/chat/completions"),
        )
        with self.assertRaisesRegex(ValueError, "credentials"):
            benchmark.endpoint_from_base("http://user:secret@localhost:30000")

    def test_portable_payload_omits_sglang_extensions_by_default(self) -> None:
        payload = benchmark.build_request_payload(
            "model", "prompt", max_tokens=32, seed=7
        )
        self.assertNotIn("ignore_eos", payload)
        self.assertNotIn("return_meta_info", payload)

        extended = benchmark.build_request_payload(
            "model",
            "prompt",
            max_tokens=32,
            seed=7,
            force_length=True,
            sglang_meta=True,
        )
        self.assertIs(extended["ignore_eos"], True)
        self.assertIs(extended["return_meta_info"], True)

    def test_weighted_wall_and_speculative_summary(self) -> None:
        runs = [
            {
                "timing": {"wall_seconds": 1.0},
                "completion_tokens": 10,
                "prompt_tokens": 3,
                "total_tokens": 13,
                "wall_tokens_per_second": 10.0,
                "output_sha256": "a" * 64,
                "finish_reason": "length",
                "meta_info": {
                    "decode_throughput": 20.0,
                    "spec_num_correct_drafts": 10,
                    "spec_num_proposed_drafts": 20,
                    "spec_verify_ct": 2,
                },
            },
            {
                "timing": {"wall_seconds": 3.0},
                "completion_tokens": 20,
                "prompt_tokens": 4,
                "total_tokens": 24,
                "wall_tokens_per_second": 20 / 3,
                "output_sha256": "b" * 64,
                "finish_reason": "length",
                "meta_info": {
                    "decode_throughput": 40.0,
                    "spec_num_correct_drafts": 5,
                    "spec_num_proposed_drafts": 10,
                    "spec_verify_ct": 3,
                },
            },
        ]
        summary = benchmark.summarize_runs(runs)
        self.assertEqual(summary["wall_tokens_per_second"], 7.5)
        self.assertEqual(summary["speculative"]["accepted_drafts_sum"], 15)
        self.assertEqual(summary["speculative"]["proposed_drafts_sum"], 30)
        self.assertEqual(summary["speculative"]["verify_cycles_sum"], 5)
        self.assertEqual(summary["speculative"]["aggregate_accept_rate"], 0.5)
        self.assertEqual(summary["speculative"]["aggregate_accept_length"], 6.0)

    def test_atomic_output_and_recursive_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            output = Path(raw_temp) / "nested" / "result.json"
            benchmark.atomic_write_json(output, {"status": "ok"})
            self.assertEqual(
                output.read_text(encoding="utf-8").strip(), '{\n  "status": "ok"\n}'
            )
        redacted = benchmark.redact_sensitive(
            {"error": "token-secret at benchmark.internal"},
            ("token-secret", "benchmark.internal"),
        )
        self.assertNotIn("token-secret", str(redacted))
        self.assertNotIn("benchmark.internal", str(redacted))


class ValidatorUnitTest(unittest.TestCase):
    def test_strict_suite_rejects_partial_or_failed_benchmark(self) -> None:
        partial = {
            "status": "failed",
            "config": {
                "normal_eos": True,
                "force_length": False,
                "cases": ["find_max"],
                "repeats_per_case": 1,
            },
            "measured_runs": [{"case": "find_max", "output": "fn x() {}"}],
        }
        with self.assertRaisesRegex(ValueError, "status"):
            validator.validate_benchmark_contract(partial, allow_partial_suite=False)
        validator.validate_benchmark_contract(partial, allow_partial_suite=True)

    def test_markdown_fence_stripping_is_conservative(self) -> None:
        code, stripped = validator.strip_markdown_rust_fence("```rust\nfn x() {}\n```")
        self.assertTrue(stripped)
        self.assertEqual(code, "fn x() {}\n")
        prose, stripped = validator.strip_markdown_rust_fence(
            "Here is code:\n```rust\nfn x() {}\n```"
        )
        self.assertFalse(stripped)
        self.assertIn("Here is code", prose)

    def test_every_exact_unique_output_is_retained(self) -> None:
        benchmark = {
            "measured_runs": [
                {"case": "find_max", "case_run": 0, "output": "fn one() {}"},
                {"case": "find_max", "case_run": 1, "output": "fn one() {}"},
                {"case": "find_max", "case_run": 2, "output": "fn two() {}"},
            ]
        }
        grouped = validator.collect_unique_outputs(benchmark)
        self.assertEqual(len(grouped["find_max"]), 2)
        self.assertEqual(grouped["find_max"][0]["occurrences"], 2)

    @unittest.skipUnless(shutil.which("rustc"), "rustc is not installed")
    def test_rustc_semantic_execution(self) -> None:
        rust_code = r"""
fn find_max(values: *const i32, len: usize) -> i32 {
    if values.is_null() || len == 0 {
        return -1;
    }
    let values = unsafe { std::slice::from_raw_parts(values, len) };
    *values.iter().max().unwrap()
}
"""
        with tempfile.TemporaryDirectory() as raw_temp:
            result = validator.compile_and_test_output(
                "find_max",
                rust_code,
                rustc=shutil.which("rustc") or "rustc",
                compile_timeout=30,
                test_timeout=10,
                temp_root=Path(raw_temp),
            )
        self.assertTrue(result["passed"], result)

    @unittest.skipUnless(shutil.which("rustc"), "rustc is not installed")
    def test_rustc_rejects_candidate_that_disables_the_harness(self) -> None:
        rust_code = "#![cfg(any())]\nfn definitely_wrong() {}\n"
        with tempfile.TemporaryDirectory() as raw_temp:
            result = validator.compile_and_test_output(
                "find_max",
                rust_code,
                rustc=shutil.which("rustc") or "rustc",
                compile_timeout=30,
                test_timeout=10,
                temp_root=Path(raw_temp),
            )
        self.assertFalse(result["passed"], result)

    @unittest.skipUnless(shutil.which("rustc"), "rustc is not installed")
    def test_rustc_rejects_candidate_that_exits_during_the_harness(self) -> None:
        rust_code = r"""
fn find_max(_values: *const i32, _len: usize) -> i32 {
    std::process::exit(0)
}
"""
        with tempfile.TemporaryDirectory() as raw_temp:
            result = validator.compile_and_test_output(
                "find_max",
                rust_code,
                rustc=shutil.which("rustc") or "rustc",
                compile_timeout=30,
                test_timeout=10,
                temp_root=Path(raw_temp),
            )
        self.assertFalse(result["passed"], result)
        self.assertEqual(result["test"]["status"], "incomplete")


class RecipeSanitizationTest(unittest.TestCase):
    def test_new_runtime_files_do_not_leak_machine_specific_values(self) -> None:
        relative_paths = (
            "docker/ominix-c2rust-hopper.Dockerfile",
            "docker/ominix-c2rust-validator.Dockerfile",
            "docs/ominix/C2RUST_QWEN35_27B_FP8_DFLASH_HOPPER.md",
            "scripts/ominix/c2rust_fp8_block128_convert.py",
            "scripts/ominix/launch_c2rust_fp8_dflash.sh",
            "scripts/ominix/benchmark_c2rust_fp8_dflash.py",
            "scripts/ominix/validate_c2rust_rust_outputs.py",
        )
        forbidden_patterns = {
            "absolute user path": re.compile(r"(?:/Users|/home)/[A-Za-z0-9._-]+/"),
            "non-loopback IPv4 address": re.compile(
                r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
            ),
            "PEM key filename": re.compile(r"(?i)\b[^\s/]+\.pem\b"),
            "private-key material": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        }
        for relative_path in relative_paths:
            path = REPO_ROOT / relative_path
            self.assertTrue(path.is_file(), relative_path)
            content = path.read_text(encoding="utf-8")
            for label, pattern in forbidden_patterns.items():
                self.assertIsNone(pattern.search(content), f"{relative_path}: {label}")


if __name__ == "__main__":
    unittest.main()
