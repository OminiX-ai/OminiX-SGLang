#!/usr/bin/env python3
"""Benchmark C2Rust FP8 + DFlash through an OpenAI-compatible endpoint.

The workload is intentionally small and reproducible: five fixed C-to-Rust
prompts, deterministic sampling, sequential requests, and round-robin case
ordering.  Normal EOS handling is the default.  SGLang-only request fields are
opt-in so the same script also works through an OminiX/OpenAI-compatible proxy.

Only the Python standard library is used.  The HTTP connection is reused across
requests, bearer credentials are read from an environment variable, and JSON
files are installed atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

PROMPTS: Dict[str, str] = {
    "find_max": (
        "Convert this C function to safe idiomatic Rust. Return only the Rust code.\n\n"
        "int find_max(const int *a, size_t n) { if (!a || n == 0) return -1; "
        "int m = a[0]; for (size_t i = 1; i < n; ++i) if (a[i] > m) m = a[i]; "
        "return m; }"
    ),
    "binary_search": (
        "Convert this C function to safe idiomatic Rust. Return only the Rust code.\n\n"
        "int binary_search(const int *a, size_t n, int needle) { size_t lo = 0, hi = n; "
        "while (lo < hi) { size_t mid = lo + (hi - lo) / 2; if (a[mid] < needle) "
        "lo = mid + 1; else hi = mid; } return lo < n && a[lo] == needle ? (int)lo : -1; }"
    ),
    "checked_sum": (
        "Convert this C function to safe idiomatic Rust. Return only the Rust code.\n\n"
        "#include <stdbool.h>\n#include <stddef.h>\n#include <stdint.h>\n\n"
        "bool sum_u32(const uint32_t *values, size_t len, uint32_t *out) { "
        "if (!values || !out) return false; uint32_t sum = 0; for (size_t i = 0; i < len; ++i) { "
        "if (UINT32_MAX - sum < values[i]) return false; sum += values[i]; } *out = sum; return true; }"
    ),
    "filter_in_place": (
        "Convert this C function to safe idiomatic Rust. Return only the Rust code.\n\n"
        "size_t remove_zeroes(int *values, size_t len) { size_t dst = 0; "
        "for (size_t src = 0; src < len; ++src) { if (values[src] != 0) "
        "values[dst++] = values[src]; } return dst; }"
    ),
    "ascii_trim": (
        "Convert this C function to safe idiomatic Rust. Return only the Rust code.\n\n"
        "void trim_ascii_spaces(const char *s, size_t len, size_t *start, size_t *end) { "
        "size_t a = 0, b = len; while (a < b && (s[a] == ' ' || s[a] == '\\t' || "
        "s[a] == '\\n' || s[a] == '\\r')) ++a; while (b > a && (s[b-1] == ' ' || "
        "s[b-1] == '\\t' || s[b-1] == '\\n' || s[b-1] == '\\r')) --b; "
        "*start = a; *end = b; }"
    ),
}

SPEC_ACCEPTED_KEYS = ("spec_accepted_drafts", "spec_num_correct_drafts")
SPEC_PROPOSED_KEYS = ("spec_proposed_drafts", "spec_num_proposed_drafts")
SPEC_VERIFY_KEYS = ("spec_verify_ct", "spec_verify_count", "spec_verify_cycles")
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp suitable for benchmark records."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def round_value(value: Optional[float]) -> Optional[float]:
    return round(value, 6) if value is not None else None


def endpoint_from_base(base_url: str) -> Tuple[str, str, int, str]:
    """Parse a base URL into connection fields without retaining credentials."""

    raw = base_url.strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("base URL must use http:// or https://")
    if not parsed.hostname:
        raise ValueError("base URL has no hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a query string or fragment")

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = path + "/chat/completions"
    else:
        endpoint_path = path + "/v1/chat/completions"
    endpoint_path = endpoint_path or "/v1/chat/completions"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname, port, endpoint_path


def request_headers(body_size: int, bearer_token: Optional[str]) -> Dict[str, str]:
    """Build request headers.  The caller must never persist the returned mapping."""

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Content-Length": str(body_size),
        "Connection": "keep-alive",
        "User-Agent": "ominix-c2rust-benchmark/1.0",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    return headers


class OpenAIChatClient:
    """Minimal persistent-connection client for JSON chat-completion requests."""

    def __init__(
        self, base_url: str, timeout_seconds: float, bearer_token: Optional[str]
    ) -> None:
        self.scheme, self.host, self.port, self.path = endpoint_from_base(base_url)
        self.timeout_seconds = timeout_seconds
        self._bearer_token = bearer_token
        self._connection: Optional[http.client.HTTPConnection] = None

    def _new_connection(self) -> http.client.HTTPConnection:
        connection_class = (
            http.client.HTTPSConnection
            if self.scheme == "https"
            else http.client.HTTPConnection
        )
        return connection_class(self.host, self.port, timeout=self.timeout_seconds)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def post_json(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if self._connection is None:
            self._connection = self._new_connection()

        try:
            self._connection.request(
                "POST",
                self.path,
                body=body,
                headers=request_headers(len(body), self._bearer_token),
            )
            response = self._connection.getresponse()
            raw_response = response.read()
        except (OSError, http.client.HTTPException):
            self.close()
            raise

        if response.getheader("Connection", "").lower() == "close":
            self.close()
        try:
            decoded = raw_response.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"server returned non-UTF-8 data (HTTP {response.status})"
            ) from exc
        try:
            data = json.loads(decoded)
        except json.JSONDecodeError as exc:
            excerpt = decoded[:1000].replace("\n", "\\n")
            raise RuntimeError(
                f"server returned invalid JSON (HTTP {response.status}): {excerpt}"
            ) from exc
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(
                f"server returned HTTP {response.status}: "
                f"{json.dumps(data, ensure_ascii=False)[:2000]}"
            )
        if not isinstance(data, dict):
            raise RuntimeError("server JSON response is not an object")
        return data


def parse_case_names(raw: str) -> List[str]:
    """Validate a comma-separated case selection while preserving its order."""

    if raw.strip().lower() == "all":
        return list(PROMPTS)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one benchmark case is required")
    unknown = [name for name in names if name not in PROMPTS]
    if unknown:
        raise ValueError(
            f"unknown case(s): {', '.join(unknown)}; available: {', '.join(PROMPTS)}"
        )
    return list(dict.fromkeys(names))


def round_robin(case_names: Sequence[str], rounds: int) -> Iterable[Tuple[int, str]]:
    """Yield one request for each case per round."""

    for round_index in range(1, rounds + 1):
        for case_name in case_names:
            yield round_index, case_name


def build_request_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    force_length: bool = False,
    sglang_meta: bool = False,
) -> Dict[str, Any]:
    """Build a portable request, adding SGLang extensions only when requested."""

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if force_length:
        payload["ignore_eos"] = True
    if sglang_meta:
        payload["return_meta_info"] = True
    return payload


def extract_message(
    response: Mapping[str, Any],
) -> Tuple[str, Optional[str], Mapping[str, Any]]:
    """Extract content, finish reason, and optional SGLang metadata."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("response does not contain choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("response does not contain choices[0].message")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise RuntimeError("choices[0].message.content is not a string")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    return content, finish_reason, meta_info


def non_negative_integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"response usage.{key} is missing or non-numeric")
    integer = int(value)
    if integer < 0 or integer != value:
        raise RuntimeError(f"response usage.{key} is not a non-negative integer")
    return integer


def finite_number(mapping: Mapping[str, Any], key: str) -> Optional[float]:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def first_finite_number(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> Optional[float]:
    for key in keys:
        value = finite_number(mapping, key)
        if value is not None:
            return value
    return None


def run_request(
    client: OpenAIChatClient,
    model: str,
    case_name: str,
    phase: str,
    case_run: int,
    max_tokens: int,
    seed: int,
    force_length: bool,
    sglang_meta: bool,
) -> Dict[str, Any]:
    """Run one request and return its complete, serializable measurement."""

    payload = build_request_payload(
        model=model,
        prompt=PROMPTS[case_name],
        max_tokens=max_tokens,
        seed=seed,
        force_length=force_length,
        sglang_meta=sglang_meta,
    )
    started_at = utc_now()
    start_ns = time.perf_counter_ns()
    response = client.post_json(payload)
    end_ns = time.perf_counter_ns()
    completed_at = utc_now()
    wall_seconds = (end_ns - start_ns) / 1_000_000_000.0

    content, finish_reason, meta_info = extract_message(response)
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("response does not contain a usage object")
    prompt_tokens = non_negative_integer(usage, "prompt_tokens")
    completion_tokens = non_negative_integer(usage, "completion_tokens")
    total_tokens = non_negative_integer(usage, "total_tokens")
    if completion_tokens and wall_seconds <= 0:
        raise RuntimeError("non-positive wall time measured for a non-empty completion")

    result: Dict[str, Any] = {
        "phase": phase,
        "case": case_name,
        "case_run": case_run,
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "wall_seconds": round(wall_seconds, 9),
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "wall_tokens_per_second": (
            round(completion_tokens / wall_seconds, 6) if completion_tokens else 0.0
        ),
        "finish_reason": finish_reason,
        "output": content,
        "output_sha256": sha256_text(content),
        "output_characters": len(content),
        "output_bytes": len(content.encode("utf-8")),
        "usage": usage,
    }
    if sglang_meta and meta_info:
        result["meta_info"] = meta_info
    return result


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate wall throughput and optional SGLang decode/spec metrics."""

    if not runs:
        return {"requests": 0}

    wall_seconds = [float(run["timing"]["wall_seconds"]) for run in runs]
    completion_tokens = [int(run["completion_tokens"]) for run in runs]
    prompt_tokens = [int(run["prompt_tokens"]) for run in runs]
    total_tokens = [int(run["total_tokens"]) for run in runs]
    per_request_tps = [float(run["wall_tokens_per_second"]) for run in runs]
    wall_seconds_sum = sum(wall_seconds)
    completion_tokens_sum = sum(completion_tokens)
    output_hashes = Counter(str(run["output_sha256"]) for run in runs)
    finish_reasons = Counter(str(run.get("finish_reason")) for run in runs)

    summary: Dict[str, Any] = {
        "requests": len(runs),
        "completion_tokens_sum": completion_tokens_sum,
        "prompt_tokens_sum": sum(prompt_tokens),
        "total_tokens_sum": sum(total_tokens),
        "wall_seconds_sum": round_value(wall_seconds_sum),
        # This aggregate is the benchmark's primary throughput number.
        "wall_tokens_per_second": round_value(
            completion_tokens_sum / wall_seconds_sum if wall_seconds_sum else None
        ),
        "request_wall_tokens_per_second_mean": round_value(
            statistics.fmean(per_request_tps)
        ),
        "request_wall_tokens_per_second_median": round_value(
            statistics.median(per_request_tps)
        ),
        "request_wall_tokens_per_second_stddev": round_value(
            statistics.stdev(per_request_tps) if len(per_request_tps) > 1 else 0.0
        ),
        "latency_seconds_mean": round_value(statistics.fmean(wall_seconds)),
        "latency_seconds_p50": round_value(percentile(wall_seconds, 0.50)),
        "latency_seconds_p95": round_value(percentile(wall_seconds, 0.95)),
        "output_hash_counts": dict(sorted(output_hashes.items())),
        "deterministic_output": len(output_hashes) == 1,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
    }

    decode_samples: List[Tuple[float, int]] = []
    complete_spec_samples: List[Tuple[float, float, float, int]] = []
    accept_rate_values: List[float] = []
    accept_length_values: List[float] = []
    spec_counter_requests = 0

    for run in runs:
        meta_info = run.get("meta_info")
        if not isinstance(meta_info, dict):
            continue
        decode_throughput = finite_number(meta_info, "decode_throughput")
        if decode_throughput is not None and decode_throughput > 0:
            decode_tokens = max(int(run["completion_tokens"]) - 1, 0)
            decode_samples.append((decode_throughput, decode_tokens))

        accepted = first_finite_number(meta_info, SPEC_ACCEPTED_KEYS)
        proposed = first_finite_number(meta_info, SPEC_PROPOSED_KEYS)
        verify = first_finite_number(meta_info, SPEC_VERIFY_KEYS)
        if accepted is not None and proposed is not None and verify is not None:
            spec_counter_requests += 1
            complete_spec_samples.append(
                (accepted, proposed, verify, int(run["completion_tokens"]))
            )
        accept_rate = finite_number(meta_info, "spec_accept_rate")
        if accept_rate is not None:
            accept_rate_values.append(accept_rate)
        accept_length = finite_number(meta_info, "spec_accept_length")
        if accept_length is not None:
            accept_length_values.append(accept_length)

    if decode_samples:
        throughput_sum = sum(sample[0] for sample in decode_samples)
        estimated_decode_seconds_sum = sum(
            tokens / throughput for throughput, tokens in decode_samples
        )
        decode_tokens_sum = sum(tokens for _, tokens in decode_samples)
        summary["decode"] = {
            "requests_with_metrics": len(decode_samples),
            "decode_tokens_sum": decode_tokens_sum,
            "decode_throughput_sum": round_value(throughput_sum),
            "decode_throughput_mean": round_value(throughput_sum / len(decode_samples)),
            "estimated_decode_seconds_sum": round_value(estimated_decode_seconds_sum),
            "aggregate_decode_tokens_per_second": round_value(
                decode_tokens_sum / estimated_decode_seconds_sum
                if estimated_decode_seconds_sum
                else None
            ),
        }

    if spec_counter_requests or accept_rate_values or accept_length_values:
        speculative: Dict[str, Any] = {
            "requests_with_counters": spec_counter_requests,
        }
        if complete_spec_samples:
            accepted_sum = sum(sample[0] for sample in complete_spec_samples)
            proposed_sum = sum(sample[1] for sample in complete_spec_samples)
            verify_cycles_sum = sum(sample[2] for sample in complete_spec_samples)
            output_tokens_sum = sum(sample[3] for sample in complete_spec_samples)
            speculative["accepted_drafts_sum"] = int(accepted_sum)
            speculative["proposed_drafts_sum"] = int(proposed_sum)
            speculative["verify_cycles_sum"] = int(verify_cycles_sum)
            if verify_cycles_sum:
                speculative["aggregate_accept_length"] = round_value(
                    output_tokens_sum / verify_cycles_sum
                )
            if proposed_sum:
                speculative["aggregate_accept_rate"] = round_value(
                    accepted_sum / proposed_sum
                )
        if accept_rate_values:
            speculative["reported_accept_rate_mean"] = round_value(
                statistics.fmean(accept_rate_values)
            )
        if accept_length_values:
            speculative["reported_accept_length_mean"] = round_value(
                statistics.fmean(accept_length_values)
            )
        summary["speculative"] = speculative

    return summary


def redact_sensitive(value: Any, sensitive_values: Sequence[Optional[str]]) -> Any:
    """Recursively redact credentials and the contacted hostname before output."""

    secrets = tuple(item for item in sensitive_values if item)
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "<redacted>")
        return value
    if isinstance(value, list):
        return [redact_sensitive(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: redact_sensitive(item, secrets) for key, item in value.items()}
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write formatted JSON beside its destination, then atomically replace it."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def emit_result(
    result: Mapping[str, Any],
    output: Optional[Path],
) -> None:
    """Atomically save when requested, then print the benchmark result."""

    if output is not None:
        atomic_write_json(output, result)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark C2Rust FP8 + DFlash through an OpenAI chat API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="C2Rust-FP8-DFlash")
    parser.add_argument(
        "--cases",
        default="all",
        help=f"comma-separated cases or 'all'; choices: {', '.join(PROMPTS)}",
    )
    parser.add_argument("--warmups", type=int, default=1, help="warmups per case")
    parser.add_argument(
        "--repeats", type=int, default=10, help="measured runs per case"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-length",
        action="store_true",
        help="send SGLang ignore_eos=true and force generation to --max-tokens",
    )
    parser.add_argument(
        "--sglang-meta",
        action="store_true",
        help="request SGLang timing and speculative-decoding metadata",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable containing an optional bearer token",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="seconds per request"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="atomically save JSON here"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress on stderr"
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="print cases and exit"
    )
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not ENVIRONMENT_NAME_RE.fullmatch(args.api_key_env):
        parser.error("--api-key-env must be a valid environment variable name")
    return args


def progress(quiet: bool, message: str) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        for name, prompt in PROMPTS.items():
            print(f"{name}\t{sha256_text(prompt)}\t{len(prompt.encode('utf-8'))} bytes")
        return 0

    bearer_token = os.environ.get(args.api_key_env) or None
    try:
        case_names = parse_case_names(args.cases)
        client = OpenAIChatClient(args.base_url, args.timeout, bearer_token)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sensitive_values = (bearer_token, client.host)
    result: Dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "ominix-c2rust-fp8-dflash",
        "started_at": utc_now(),
        "config": {
            "model": args.model,
            "cases": case_names,
            "warmups_per_case": args.warmups,
            "repeats_per_case": args.repeats,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "normal_eos": not args.force_length,
            "force_length": args.force_length,
            "sglang_meta": args.sglang_meta,
            "stream": False,
            "timeout_seconds": args.timeout,
            "authenticated": bool(bearer_token),
        },
        "prompts": {
            name: {
                "text": PROMPTS[name],
                "sha256": sha256_text(PROMPTS[name]),
                "bytes": len(PROMPTS[name].encode("utf-8")),
            }
            for name in case_names
        },
        "warmup_runs": [],
        "measured_runs": [],
    }

    total_warmups = args.warmups * len(case_names)
    total_measured = args.repeats * len(case_names)
    progress(
        args.quiet,
        f"model={args.model} warmups={total_warmups} measured={total_measured}",
    )
    try:
        for round_index, case_name in round_robin(case_names, args.warmups):
            progress(
                args.quiet,
                f"warmup {round_index}/{args.warmups}: {case_name}",
            )
            run = run_request(
                client,
                args.model,
                case_name,
                "warmup",
                round_index,
                args.max_tokens,
                args.seed,
                args.force_length,
                args.sglang_meta,
            )
            result["warmup_runs"].append(run)
            progress(
                args.quiet,
                f"  {run['completion_tokens']} tokens in "
                f"{run['timing']['wall_seconds']:.3f}s = "
                f"{run['wall_tokens_per_second']:.3f} tok/s",
            )

        for round_index, case_name in round_robin(case_names, args.repeats):
            progress(
                args.quiet,
                f"measured {round_index}/{args.repeats}: {case_name}",
            )
            run = run_request(
                client,
                args.model,
                case_name,
                "measured",
                round_index,
                args.max_tokens,
                args.seed,
                args.force_length,
                args.sglang_meta,
            )
            result["measured_runs"].append(run)
            progress(
                args.quiet,
                f"  {run['completion_tokens']} tokens in "
                f"{run['timing']['wall_seconds']:.3f}s = "
                f"{run['wall_tokens_per_second']:.3f} tok/s "
                f"sha256={run['output_sha256'][:12]}",
            )
    except (OSError, http.client.HTTPException, RuntimeError) as exc:
        safe_message = redact_sensitive(str(exc), sensitive_values)
        result["status"] = "failed"
        result["completed_at"] = utc_now()
        result["error"] = {"type": type(exc).__name__, "message": safe_message}
        emit_result(result, args.output)
        print(f"benchmark failed: {safe_message}", file=sys.stderr)
        return 1
    finally:
        client.close()

    measured_runs: List[Mapping[str, Any]] = result["measured_runs"]
    result["summary"] = {
        "overall": summarize_runs(measured_runs),
        "by_case": {
            case_name: summarize_runs(
                [run for run in measured_runs if run["case"] == case_name]
            )
            for case_name in case_names
        },
    }
    result["status"] = "ok"
    result["completed_at"] = utc_now()
    emit_result(result, args.output)
    overall = result["summary"]["overall"]
    progress(
        args.quiet,
        f"aggregate wall throughput: {overall['wall_tokens_per_second']:.3f} tok/s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
