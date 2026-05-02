# OminiX Worker V0 Tokenizer And Detokenizer Boundary

This slice keeps public OpenAI JSON outside `sglang-ominix` while making the
worker boundary responsible for prompt tokenization and generated-token
detokenization.

## Helper

The dependency-light helper lives at:

```text
scripts/ominix/worker_token_boundary.py
```

It exposes:

```python
prepared = prepare_generate_request(
    request_json,
    tokenizer=tokenizer,
    chat_template=None,
)
```

The return value is:

```python
PreparedGenerate(
    request_id=str,
    original_text=str,
    input_ids=list[int],
    sampling_params=dict,
    stream=bool,
)
```

OSO-013 should call this before constructing the gRPC
`SglangScheduler.Generate` request. `prepared.input_ids` maps to the scheduler
tokenized input. `prepared.sampling_params` maps to the current SGLang sampling
proto fields. The helper normalizes OminiX `max_completion_tokens` and legacy
`max_tokens` to SGLang `max_new_tokens` when `max_new_tokens` is absent.

## Accepted Inputs

`GenerateRequest.input.kind == "tokens"` bypasses tokenizer lookup:

```json
{
  "input": {"kind": "tokens", "tokens": [1, 2, 3]},
  "sampling": {"max_completion_tokens": 16},
  "stream": true
}
```

`input.kind == "text"` and `input.kind == "completion"` require an injected
tokenizer with `encode(text)`.

`input.kind == "chat"` requires the real tokenizer from the launched model. The
helper calls:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
    **chat_template_kwargs,
)
```

If a separate template object is injected for tests or future SGLang template
manager integration, its `apply_chat_template(...)` method is called with the
same arguments. The helper does not hardcode a DeepSeek prompt template.

For `deepseek-ai/DeepSeek-V4-Flash`, the production dependency is the tokenizer
and chat template loaded from the model/tokenizer config in the normal SGLang
launch path. If no tokenizer is injected for chat/text input, the helper raises
`MissingTokenizerError` instead of silently accepting raw text.

## Detokenization

`TokenDeltaDecoder` accepts generated token chunks and emits incremental text:

```python
decoder = TokenDeltaDecoder(
    tokenizer,
    stop_strings=sampling_params.get("stop"),
    stop_token_ids=sampling_params.get("stop_token_ids"),
    skip_special_tokens=True,
    spaces_between_special_tokens=True,
)

text_deltas = decoder.accept([token_id])
final_delta = decoder.finish()
```

The decoder uses `tokenizer.decode(ids, skip_special_tokens=...,
spaces_between_special_tokens=...)`. If a tokenizer does not accept
`spaces_between_special_tokens`, the helper retries without that argument.

Stop-token IDs are not decoded. Stop strings are held back by up to the longest
stop string minus one character so streaming output can truncate a stop string
that spans token chunks.

## Local Coverage

`scripts/ominix/test_worker_token_boundary.py` covers:

- token passthrough without tokenizer
- completion text with an injected stub tokenizer
- chat with an injected stub template/tokenizer
- chat dependency error when tokenizer is absent
- incremental token-delta decoding
- stop string truncation
- stop token truncation
