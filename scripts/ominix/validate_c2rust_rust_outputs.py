#!/usr/bin/env python3
"""Compile and execute semantic tests for C2Rust benchmark outputs.

The validator checks every distinct output in ``measured_runs``.  Reference
results are deliberately informational: a content-hash mismatch never changes
the validation status because two correct Rust implementations need not be
byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

TESTS: dict[str, str] = {
    "find_max": r"""
pub(super) fn __ominix_validator_behavior() {
    let values = [-2, 7, 3];
    ::std::assert_eq!(find_max(values.as_ptr(), values.len()), 7);
    ::std::assert_eq!(find_max(::std::ptr::null(), values.len()), -1);
    ::std::assert_eq!(find_max(values.as_ptr(), 0), -1);
}
""",
    "binary_search": r"""
pub(super) fn __ominix_validator_behavior() {
    let values = [-2, 0, 4, 4, 9];
    ::std::assert_eq!(binary_search(&values, -2), 0);
    ::std::assert_eq!(binary_search(&values, 4), 2);
    ::std::assert_eq!(binary_search(&values, 9), 4);
    ::std::assert_eq!(binary_search(&values, 5), -1);
    ::std::assert_eq!(binary_search(&[], 1), -1);
}
""",
    "checked_sum": r"""
pub(super) fn __ominix_validator_behavior() {
    let values = [1_u32, 2, 3];
    let mut out = 99;
    ::std::assert!(sum_u32(values.as_ptr(), values.len(), &mut out));
    ::std::assert_eq!(out, 6);

    let overflow = [u32::MAX, 1];
    out = 77;
    ::std::assert!(!sum_u32(overflow.as_ptr(), overflow.len(), &mut out));
    ::std::assert_eq!(out, 77);
    ::std::assert!(!sum_u32(::std::ptr::null(), 1, &mut out));
    ::std::assert!(!sum_u32(
        values.as_ptr(),
        values.len(),
        ::std::ptr::null_mut(),
    ));
}
""",
    "filter_in_place": r"""
pub(super) fn __ominix_validator_behavior() {
    let mut values = [0, 1, 0, 2, 3, 0];
    let n = remove_zeroes(&mut values);
    ::std::assert_eq!(n, 3);
    ::std::assert_eq!(&values[..n], &[1, 2, 3]);

    let mut zeroes = [0, 0];
    ::std::assert_eq!(remove_zeroes(&mut zeroes), 0);
    ::std::assert_eq!(remove_zeroes(&mut []), 0);
}
""",
    "ascii_trim": r"""
pub(super) fn __ominix_validator_behavior() {
    let value = b" \tfoo\r\n";
    let (mut start, mut end) = (99, 99);
    trim_ascii_spaces(
        value.as_ptr() as *const i8,
        value.len(),
        &mut start,
        &mut end,
    );
    ::std::assert_eq!((start, end), (2, 5));

    let spaces = b"   ";
    trim_ascii_spaces(
        spaces.as_ptr() as *const i8,
        spaces.len(),
        &mut start,
        &mut end,
    );
    ::std::assert_eq!((start, end), (3, 3));
}
""",
}

VALIDATOR_TEST_NAME = "__ominix_validator_tests::behavior"


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_markdown_rust_fence(value: str) -> tuple[str, bool]:
    """Remove one enclosing Markdown Rust (or unlabelled) code fence.

    Text outside the fence is intentionally not discarded.  A response that
    contains prose as well as code should fail compilation rather than having
    part of its content silently ignored.
    """

    stripped = value.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2:
        opening = lines[0].strip().lower()
        closing = lines[-1].strip()
        if opening in {"```", "```rust", "```rs"} and closing == "```":
            return "\n".join(lines[1:-1]).strip() + "\n", True
    return stripped + ("\n" if stripped else ""), False


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and require an object at its root."""

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("benchmark JSON root must be an object")
    return value


def collect_unique_outputs(
    benchmark: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Group every unique measured output by benchmark case.

    Uniqueness is based on the exact response content, before optional Markdown
    fences are removed.  Returned entries are JSON-serializable and ordered by
    first appearance within each case.
    """

    runs = benchmark.get("measured_runs")
    if not isinstance(runs, list):
        raise ValueError("benchmark JSON must contain a measured_runs array")
    if not runs:
        raise ValueError("benchmark JSON contains no measured runs")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run_index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise ValueError(f"measured_runs[{run_index}] must be an object")
        case = run.get("case")
        output = run.get("output")
        if not isinstance(case, str) or not case:
            raise ValueError(f"measured_runs[{run_index}].case must be a string")
        if not isinstance(output, str):
            raise ValueError(
                f"measured_runs[{run_index}].output must be a string; "
                "rerun the benchmark without content omission"
            )

        entry = grouped[case].get(output)
        if entry is None:
            normalized, fence_stripped = strip_markdown_rust_fence(output)
            entry = {
                "content": output,
                "content_sha256": sha256_text(output),
                "normalized_content": normalized,
                "normalized_sha256": sha256_text(normalized),
                "markdown_fence_stripped": fence_stripped,
                "occurrences": 0,
                "run_indices": [],
                "case_runs": [],
            }
            grouped[case][output] = entry
        entry["occurrences"] += 1
        entry["run_indices"].append(run_index)
        if isinstance(run.get("case_run"), int):
            entry["case_runs"].append(run["case_run"])

    return {case: list(entries.values()) for case, entries in grouped.items()}


def collect_case_content_hashes(
    benchmark: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Collect exact response-content hashes, including content-omitted results."""

    runs = benchmark.get("measured_runs")
    if not isinstance(runs, list):
        raise ValueError("reference JSON must contain a measured_runs array")

    hashes: dict[str, set[str]] = defaultdict(set)
    for run_index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise ValueError(f"reference measured_runs[{run_index}] must be an object")
        case = run.get("case")
        if not isinstance(case, str) or not case:
            raise ValueError(
                f"reference measured_runs[{run_index}].case must be a string"
            )
        output = run.get("output")
        recorded_hash = run.get("output_sha256")
        if isinstance(output, str):
            hashes[case].add(sha256_text(output))
        elif isinstance(recorded_hash, str) and re.fullmatch(
            r"[0-9a-fA-F]{64}", recorded_hash
        ):
            hashes[case].add(recorded_hash.lower())
        else:
            raise ValueError(
                f"reference measured_runs[{run_index}] has neither output content "
                "nor an output_sha256 digest"
            )
    return dict(hashes)


def _clean_process_text(value: str | bytes | None, temp_root: Path) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    cleaned = value.replace(str(temp_root), "<temp>")
    limit = 16_000
    if len(cleaned) > limit:
        return cleaned[:limit] + f"\n... truncated {len(cleaned) - limit} characters"
    return cleaned


def _run_process(
    command: Sequence[str],
    *,
    timeout: float,
    temp_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    process_env = {
        "HOME": str(temp_root),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": str(temp_root),
    }
    for name in ("CARGO_HOME", "LD_LIBRARY_PATH", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN"):
        if os.environ.get(name):
            process_env[name] = os.environ[name]
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=temp_root) as out:
            with tempfile.TemporaryFile(
                mode="w+", encoding="utf-8", dir=temp_root
            ) as err:
                process = subprocess.Popen(
                    command,
                    cwd=temp_root,
                    env=process_env,
                    stdout=out,
                    stderr=err,
                    text=True,
                    start_new_session=os.name == "posix",
                )
                timed_out = False
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                out.seek(0)
                err.seek(0)
                stdout = out.read()
                stderr = err.read()
    except OSError as exc:
        return {
            "status": "error",
            "passed": False,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "error": str(exc),
        }

    if timed_out:
        return {
            "status": "timeout",
            "passed": False,
            "timeout_seconds": timeout,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "stdout": _clean_process_text(stdout, temp_root),
            "stderr": _clean_process_text(stderr, temp_root),
        }

    return {
        "status": "passed" if process.returncode == 0 else "failed",
        "passed": process.returncode == 0,
        "returncode": process.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "stdout": _clean_process_text(stdout, temp_root),
        "stderr": _clean_process_text(stderr, temp_root),
    }


def validate_benchmark_contract(
    benchmark: Mapping[str, Any], *, allow_partial_suite: bool
) -> None:
    """Require a successful five-case normal-EOS benchmark by default."""
    if allow_partial_suite:
        return
    if benchmark.get("status") != "ok":
        raise ValueError("benchmark status must be 'ok'")
    config = benchmark.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("benchmark config must be an object")
    if config.get("normal_eos") is not True or config.get("force_length") is not False:
        raise ValueError("semantic validation requires a normal-EOS benchmark")
    cases = config.get("cases")
    if (
        not isinstance(cases, list)
        or set(cases) != set(TESTS)
        or len(cases) != len(TESTS)
    ):
        raise ValueError(
            "semantic validation requires all five C2Rust cases exactly once"
        )
    repeats = config.get("repeats_per_case")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("benchmark repeats_per_case must be a positive integer")
    runs = benchmark.get("measured_runs")
    if not isinstance(runs, list):
        raise ValueError("benchmark JSON must contain a measured_runs array")
    counts = {case: 0 for case in TESTS}
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or run.get("case") not in counts:
            raise ValueError(f"measured_runs[{index}] has an unexpected case")
        counts[str(run["case"])] += 1
    wrong_counts = {case: count for case, count in counts.items() if count != repeats}
    if wrong_counts:
        raise ValueError(
            f"measured run counts do not match repeats_per_case={repeats}: {wrong_counts}"
        )


def compile_and_test_output(
    case: str,
    rust_code: str,
    *,
    rustc: str,
    compile_timeout: float,
    test_timeout: float,
    temp_root: Path,
    output_index: int = 0,
) -> dict[str, Any]:
    """Compile one answer with its semantic test and execute the test binary."""

    if case not in TESTS:
        return {
            "passed": False,
            "error": f"unsupported benchmark case: {case}",
            "compile": {"status": "not_run", "passed": False},
            "test": {"status": "not_run", "passed": False},
        }

    safe_case = "".join(char if char.isalnum() else "_" for char in case)
    source_path = temp_root / f"{safe_case}-{output_index}.rs"
    candidate_path = temp_root / f"{safe_case}-{output_index}-candidate.rs"
    adapter_path = temp_root / f"{safe_case}-{output_index}-adapter.rs"
    binary_path = temp_root / f"{safe_case}-{output_index}-test"
    candidate_path.write_text(rust_code.rstrip() + "\n", encoding="utf-8")
    adapter_path.write_text(TESTS[case].lstrip(), encoding="utf-8")
    source_path.write_text(
        f"""mod __ominix_candidate {{
    include!({json.dumps(candidate_path.name)});
    include!({json.dumps(adapter_path.name)});
}}

#[cfg(test)]
mod __ominix_validator_tests {{
    #[test]
    fn behavior() {{
        super::__ominix_candidate::__ominix_validator_behavior();
    }}
}}
""",
        encoding="utf-8",
    )

    compile_result = _run_process(
        [
            rustc,
            "--edition=2021",
            "--test",
            str(source_path),
            "-o",
            str(binary_path),
        ],
        timeout=compile_timeout,
        temp_root=temp_root,
    )
    if not compile_result["passed"]:
        return {
            "passed": False,
            "compile": compile_result,
            "list": {"status": "not_run", "passed": False},
            "test": {"status": "not_run", "passed": False},
        }

    list_result = _run_process(
        [str(binary_path), "--list", "--format", "terse"],
        timeout=test_timeout,
        temp_root=temp_root,
    )
    listed_tests = [
        line.removesuffix(": test")
        for line in list_result.get("stdout", "").splitlines()
        if line.endswith(": test")
    ]
    if not list_result["passed"] or listed_tests.count(VALIDATOR_TEST_NAME) != 1:
        list_result["passed"] = False
        list_result["expected_test"] = VALIDATOR_TEST_NAME
        list_result["listed_tests"] = listed_tests
        return {
            "passed": False,
            "compile": compile_result,
            "list": list_result,
            "test": {"status": "not_run", "passed": False},
        }

    test_result = _run_process(
        [str(binary_path), "--exact", VALIDATOR_TEST_NAME],
        timeout=test_timeout,
        temp_root=temp_root,
    )
    test_lines = test_result.get("stdout", "").splitlines()
    success_line = f"test {VALIDATOR_TEST_NAME} ... ok"
    success_summary = (
        "test result: ok. 1 passed; 0 failed; 0 ignored; " "0 measured; 0 filtered out;"
    )
    completed = success_line in test_lines and any(
        line.startswith(success_summary) for line in test_lines
    )
    if test_result["passed"] and not completed:
        test_result["passed"] = False
        test_result["status"] = "incomplete"
        test_result["error"] = (
            "semantic test process exited without libtest reporting one completed test"
        )
    return {
        "passed": bool(test_result["passed"]),
        "compile": compile_result,
        "list": list_result,
        "test": test_result,
    }


def validate_benchmark(
    benchmark: Mapping[str, Any],
    *,
    rustc: str,
    compile_timeout: float,
    test_timeout: float,
    reference: Optional[Mapping[str, Any]] = None,
    allow_partial_suite: bool = False,
) -> dict[str, Any]:
    """Validate all distinct outputs and return a JSON-serializable summary."""

    validate_benchmark_contract(benchmark, allow_partial_suite=allow_partial_suite)
    outputs_by_case = collect_unique_outputs(benchmark)
    reference_hashes = (
        collect_case_content_hashes(reference) if reference is not None else None
    )

    case_results: dict[str, Any] = {}
    total_outputs = 0
    passed_outputs = 0
    with tempfile.TemporaryDirectory(prefix="c2rust-rust-validation-") as raw_temp:
        temp_root = Path(raw_temp)
        for case in sorted(outputs_by_case):
            output_results: list[dict[str, Any]] = []
            for output_index, entry in enumerate(outputs_by_case[case]):
                validation = compile_and_test_output(
                    case,
                    entry["normalized_content"],
                    rustc=rustc,
                    compile_timeout=compile_timeout,
                    test_timeout=test_timeout,
                    temp_root=temp_root,
                    output_index=output_index,
                )
                output_result = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"content", "normalized_content"}
                }
                output_result.update(validation)
                if reference_hashes is not None:
                    output_result["reference_content_hash_match"] = entry[
                        "content_sha256"
                    ] in reference_hashes.get(case, set())
                output_results.append(output_result)

            unique_count = len(output_results)
            passed_count = sum(bool(item["passed"]) for item in output_results)
            run_count = sum(int(item["occurrences"]) for item in output_results)
            case_result: dict[str, Any] = {
                "passed": unique_count > 0 and passed_count == unique_count,
                "measured_runs": run_count,
                "unique_outputs": unique_count,
                "passed_outputs": passed_count,
                "outputs": output_results,
            }
            if reference_hashes is not None:
                candidate_set = {str(item["content_sha256"]) for item in output_results}
                reference_set = reference_hashes.get(case, set())
                case_result["reference"] = {
                    "content_hash_match": candidate_set == reference_set,
                    "candidate_content_sha256": sorted(candidate_set),
                    "reference_content_sha256": sorted(reference_set),
                }
            case_results[case] = case_result
            total_outputs += unique_count
            passed_outputs += passed_count

    all_passed = total_outputs > 0 and passed_outputs == total_outputs
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "passed": all_passed,
        "summary": {
            "cases": len(case_results),
            "passed_cases": sum(
                bool(case_result["passed"]) for case_result in case_results.values()
            ),
            "unique_outputs": total_outputs,
            "passed_outputs": passed_outputs,
        },
        "cases": case_results,
    }
    if reference_hashes is not None:
        compared_cases = set(case_results) | set(reference_hashes)
        result["reference"] = {
            "provided": True,
            "content_hash_match": all(
                set(
                    item["content_sha256"]
                    for item in case_results.get(case, {}).get("outputs", [])
                )
                == reference_hashes.get(case, set())
                for case in compared_cases
            ),
            "affects_validation_status": False,
        }
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and semantically test every unique C2Rust benchmark output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="optional benchmark whose exact content hashes are reported for comparison",
    )
    parser.add_argument("--rustc", default="rustc", help="Rust compiler executable")
    parser.add_argument(
        "--compile-timeout", type=float, default=60.0, help="seconds per compilation"
    )
    parser.add_argument(
        "--test-timeout", type=float, default=30.0, help="seconds per test binary"
    )
    parser.add_argument(
        "--output", type=Path, help="also write the JSON summary to this file"
    )
    parser.add_argument(
        "--allow-partial-suite",
        action="store_true",
        help="allow successful subsets/nonstandard benchmarks for diagnostics",
    )
    parser.add_argument(
        "--execute-generated-code",
        action="store_true",
        help="acknowledge that generated Rust will be compiled and executed",
    )
    return parser.parse_args(argv)


def _atomic_write(path: Path, rendered: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _emit_json(result: Mapping[str, Any], output: Optional[Path]) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    if output is not None:
        _atomic_write(output, rendered)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.compile_timeout <= 0 or args.test_timeout <= 0:
        result = {
            "schema_version": 1,
            "status": "error",
            "passed": False,
            "error": "timeouts must be positive",
        }
        _emit_json(result, args.output)
        return 2
    if not args.execute_generated_code:
        result = {
            "schema_version": 1,
            "status": "error",
            "passed": False,
            "error": (
                "refusing to execute model-generated code without "
                "--execute-generated-code; use the documented locked-down container"
            ),
        }
        _emit_json(result, args.output)
        return 2

    rustc = shutil.which(args.rustc)
    if rustc is None:
        result = {
            "schema_version": 1,
            "status": "error",
            "passed": False,
            "error": f"Rust compiler not found: {args.rustc}",
        }
        _emit_json(result, args.output)
        return 2

    try:
        benchmark = load_json_object(args.benchmark_json)
        reference = (
            load_json_object(args.reference_json)
            if args.reference_json is not None
            else None
        )
        result = validate_benchmark(
            benchmark,
            rustc=rustc,
            compile_timeout=args.compile_timeout,
            test_timeout=args.test_timeout,
            reference=reference,
            allow_partial_suite=args.allow_partial_suite,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": 1,
            "status": "error",
            "passed": False,
            "error": str(exc),
        }
        _emit_json(result, args.output)
        return 2

    _emit_json(result, args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
