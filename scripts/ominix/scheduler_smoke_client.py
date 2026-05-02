#!/usr/bin/env python3
"""Current SGLang-shape tokenized gRPC smoke client for OminiX parity work."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import uuid


CURRENT_SGLANG_PACKAGE = "sglang.grpc.scheduler"


def _load_proto_modules():
    modules = {}
    missing = []
    for module_name in (
        "grpc",
        "google.protobuf.json_format",
        "smg_grpc_proto.sglang_scheduler_pb2",
        "smg_grpc_proto.sglang_scheduler_pb2_grpc",
    ):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing.append(exc.name or module_name)

    if missing:
        print(
            "missing dependency modules: "
            + ", ".join(sorted(set(missing)))
            + "\ninstall the sglang-ominix Python package or at least "
            "grpcio and smg-grpc-proto before running this client",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return (
        modules["grpc"],
        modules["google.protobuf.json_format"].MessageToDict,
        modules["smg_grpc_proto.sglang_scheduler_pb2"],
        modules["smg_grpc_proto.sglang_scheduler_pb2_grpc"],
    )


def _parse_ids(raw: str) -> list[int]:
    try:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--input-ids must be a comma-separated integer list"
        ) from exc


def _print_message(label: str, message, to_dict):
    print(f"\n== {label} ==")
    print(to_dict(message, preserving_proto_field_name=True))


def _field_names(message_cls) -> set[str]:
    return {field.name for field in message_cls.DESCRIPTOR.fields}


def _require_fields(pb2, message_name: str, fields: tuple[str, ...]) -> list[str]:
    message_cls = getattr(pb2, message_name, None)
    if message_cls is None:
        return [f"missing message {message_name}"]

    available = _field_names(message_cls)
    return [
        f"missing field {message_name}.{field}"
        for field in fields
        if field not in available
    ]


def _require_presence(message, field_name: str) -> str | None:
    try:
        message.HasField(field_name)
    except ValueError:
        return (
            f"{message.DESCRIPTOR.full_name}.{field_name} lacks field presence; "
            "grpc_server.py calls HasField for this field"
        )
    return None


def _build_generate_request(pb2, args, request_id: str):
    # Explicit defaults avoid proto3 zero-values that do not match SGLang's
    # semantic sampling defaults.
    return pb2.GenerateRequest(
        request_id=request_id,
        tokenized=pb2.TokenizedInput(
            input_ids=args.input_ids,
            original_text=args.original_text,
        ),
        sampling_params=pb2.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=args.max_new_tokens,
            n=1,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        ),
        stream=True,
    )


def _check_proto_only(pb2, pb2_grpc) -> int:
    errors = []
    warnings = []

    package = getattr(pb2.DESCRIPTOR, "package", "")
    if package != CURRENT_SGLANG_PACKAGE:
        errors.append(
            f"descriptor package is {package!r}, expected {CURRENT_SGLANG_PACKAGE!r}"
        )

    service = pb2.DESCRIPTOR.services_by_name.get("SglangScheduler")
    if service is None:
        errors.append("missing service SglangScheduler")
        methods = set()
    else:
        methods = {method.name for method in service.methods}
        print(f"service: {service.full_name}")

    for method_name in ("Generate", "Abort", "HealthCheck", "GetModelInfo", "GetLoads"):
        if method_name not in methods:
            errors.append(f"missing service method SglangScheduler.{method_name}")

    for grpc_symbol in (
        "SglangSchedulerStub",
        "SglangSchedulerServicer",
        "add_SglangSchedulerServicer_to_server",
    ):
        if not hasattr(pb2_grpc, grpc_symbol):
            errors.append(f"missing generated grpc symbol {grpc_symbol}")

    required_fields = {
        "GenerateRequest": (
            "request_id",
            "tokenized",
            "sampling_params",
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "token_ids_logprob",
            "disaggregated_params",
            "lora_id",
            "stream",
        ),
        "TokenizedInput": ("original_text", "input_ids"),
        "SamplingParams": (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "max_new_tokens",
            "stop",
            "stop_token_ids",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "regex",
            "json_schema",
            "ebnf_grammar",
            "structural_tag",
            "n",
            "min_new_tokens",
            "ignore_eos",
            "no_stop_trim",
            "stream_interval",
            "logit_bias",
            "custom_params",
        ),
        "DisaggregatedParams": (
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
        ),
        "GenerateResponse": ("request_id", "chunk", "complete", "error"),
        "GenerateStreamChunk": (
            "token_ids",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "output_logprobs",
            "input_logprobs",
            "index",
        ),
        "GenerateComplete": (
            "output_ids",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "output_logprobs",
            "input_logprobs",
            "index",
        ),
        "AbortRequest": ("request_id", "reason"),
        "HealthCheckRequest": (),
        "GetModelInfoRequest": (),
        "GetModelInfoResponse": (
            "model_path",
            "tokenizer_path",
            "is_generation",
            "preferred_sampling_params",
            "weight_version",
            "served_model_name",
            "max_context_length",
            "vocab_size",
            "supports_vision",
            "model_type",
            "architectures",
            "eos_token_ids",
            "pad_token_id",
            "bos_token_id",
            "max_req_input_len",
            "id2label_json",
            "num_labels",
        ),
        "GetLoadsRequest": ("include", "dp_rank"),
        "GetLoadsResponse": (
            "timestamp",
            "version",
            "dp_rank_count",
            "loads",
            "aggregate",
        ),
        "SchedulerLoad": (
            "dp_rank",
            "num_running_reqs",
            "num_waiting_reqs",
            "num_total_reqs",
            "num_used_tokens",
            "max_total_num_tokens",
            "token_usage",
            "gen_throughput",
            "cache_hit_rate",
            "utilization",
            "max_running_requests",
            "memory",
            "speculative",
            "lora",
            "disaggregation",
            "queues",
        ),
        "AggregateMetrics": (
            "total_running_reqs",
            "total_waiting_reqs",
            "total_reqs",
            "avg_token_usage",
            "avg_throughput",
            "avg_utilization",
        ),
    }
    for message_name, fields in required_fields.items():
        errors.extend(_require_fields(pb2, message_name, fields))

    if hasattr(pb2, "SamplingParams"):
        sampling_params = pb2.SamplingParams()
        for field_name in ("max_new_tokens", "stream_interval"):
            presence_error = _require_presence(sampling_params, field_name)
            if presence_error:
                errors.append(presence_error)

    if hasattr(pb2, "GetLoadsRequest"):
        loads_request = pb2.GetLoadsRequest()
        presence_error = _require_presence(loads_request, "dp_rank")
        if presence_error:
            errors.append(presence_error)

    try:
        request = _build_generate_request(
            pb2,
            argparse.Namespace(
                input_ids=[1, 2, 3],
                original_text="tokenized smoke",
                max_new_tokens=1,
            ),
            "ominix-proto-check",
        )
        if not request.HasField("tokenized"):
            errors.append("constructed GenerateRequest does not set tokenized")
        if not request.sampling_params.HasField("max_new_tokens"):
            errors.append("constructed SamplingParams does not set max_new_tokens")
    except Exception as exc:
        errors.append(f"failed to construct minimal GenerateRequest: {exc}")

    print(f"package: {package or '<missing>'}")
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nproto check: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nproto check: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current SGLang gRPC Generate/Abort/Health/GetLoads/"
            "GetModelInfo smoke path. This is not the canonical OminiX v0 "
            "TokenizedGenerate envelope yet."
        )
    )
    parser.add_argument("--target", default="127.0.0.1:30000")
    parser.add_argument("--input-ids", type=_parse_ids, default=[1, 2, 3])
    parser.add_argument("--original-text", default="tokenized smoke")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--abort-after-first-chunk", action="store_true")
    parser.add_argument(
        "--check-proto-only",
        action="store_true",
        help="validate installed Python gRPC/proto modules without contacting a server",
    )
    args = parser.parse_args()

    grpc, message_to_dict, pb2, pb2_grpc = _load_proto_modules()
    if args.check_proto_only:
        return _check_proto_only(pb2, pb2_grpc)

    channel = grpc.insecure_channel(
        args.target,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )
    stub = pb2_grpc.SglangSchedulerStub(channel)

    try:
        health = stub.HealthCheck(pb2.HealthCheckRequest(), timeout=args.timeout)
        _print_message("HealthCheck", health, message_to_dict)

        model_info = stub.GetModelInfo(pb2.GetModelInfoRequest(), timeout=args.timeout)
        _print_message("GetModelInfo", model_info, message_to_dict)

        if hasattr(stub, "GetLoads") and hasattr(pb2, "GetLoadsRequest"):
            loads = stub.GetLoads(pb2.GetLoadsRequest(include=["all"]), timeout=args.timeout)
            _print_message("GetLoads", loads, message_to_dict)
        else:
            print(
                "\n== GetLoads ==\nSKIP: installed smg_grpc_proto does not expose GetLoads"
            )

        request_id = f"ominix-smoke-{uuid.uuid4().hex}"
        request = _build_generate_request(pb2, args, request_id)

        print(f"\n== Generate request_id={request_id} ==")
        stream = stub.Generate(request, timeout=args.timeout)
        saw_first_chunk = False
        for response in stream:
            _print_message("GenerateResponse", response, message_to_dict)
            if args.abort_after_first_chunk and not saw_first_chunk:
                saw_first_chunk = True
                abort = stub.Abort(
                    pb2.AbortRequest(
                        request_id=request_id,
                        reason=f"smoke abort after first chunk at {time.time()}",
                    ),
                    timeout=args.timeout,
                )
                _print_message("Abort", abort, message_to_dict)
    finally:
        channel.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
