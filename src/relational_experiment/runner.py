from __future__ import annotations

import hashlib
import json
import os
import re
import time
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .io import append_jsonl, read_jsonl
from .model import Trial

STRICT_ANSWER = re.compile(r"^\s*([A-D])\s*[.!]?\s*$", re.IGNORECASE)
LENIENT_ANSWER = re.compile(r"(?:answer(?:\s+is)?\s*[:=]?\s*)?\b([A-D])\b", re.IGNORECASE)
FINAL_PREFIX = re.compile(
    r"^\s*(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*[:=]\s*",
    re.IGNORECASE,
)


def configuration_fingerprint(config: dict) -> str:
    stable = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


def trial_fingerprint(trial: Trial) -> str:
    stable = json.dumps(trial.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


def parse_answer(content: str) -> tuple[str | None, str]:
    strict = STRICT_ANSWER.fullmatch(content)
    if strict:
        return strict.group(1).upper(), "strict"
    matches = {match.upper() for match in LENIENT_ANSWER.findall(content)}
    if len(matches) == 1:
        return matches.pop(), "lenient"
    return None, "unparseable"


def parse_trial_answer(content: str, trial: Trial) -> tuple[str | None, str]:
    if not trial.accepted_answers:
        return parse_answer(content)
    normalized = content.strip().lower().rstrip(".! ")
    allowed = {answer.lower(): answer for answer in trial.accepted_answers}
    if normalized in allowed:
        return allowed[normalized], "strict"
    return None, "unparseable"


def parse_structured_trial_answer(content: str, trial: Trial) -> tuple[str | None, str]:
    """Parse the schema-constrained {"answer": ...} response without reading reasoning."""
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None, "structured-invalid-json"
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None, "structured-invalid-shape"
    answer, mode = parse_trial_answer(str(value["answer"]), trial)
    return answer, f"structured-{mode}"


def split_reasoning_output(
    content: str, reasoning_content: str | None = None
) -> tuple[str, str, str]:
    """Return (reasoning, final line, split mode) without searching the trace."""
    if reasoning_content is not None:
        final_lines = [line.strip() for line in content.splitlines() if line.strip()]
        return reasoning_content.strip(), final_lines[-1] if final_lines else "", "separate-field"
    if "</think>" in content:
        reasoning, final = content.rsplit("</think>", 1)
        reasoning = reasoning.replace("<think>", "", 1).strip()
        final_lines = [line.strip() for line in final.splitlines() if line.strip()]
        return reasoning, final_lines[-1] if final_lines else "", "think-tags"
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "", "", "empty"
    return "\n".join(lines[:-1]), lines[-1], "last-line"


def parse_reasoning_trial_answer(
    content: str,
    trial: Trial,
    reasoning_content: str | None = None,
    truncated: bool = False,
) -> tuple[str | None, str, str, str, str]:
    """Parse only the explicit final line; truncated deliberations are always invalid."""
    reasoning, final_line, split_mode = split_reasoning_output(content, reasoning_content)
    candidate = FINAL_PREFIX.sub("", final_line).strip()
    candidate = candidate.strip("`*_ ")
    if truncated:
        return None, "truncated", reasoning, final_line, split_mode
    answer, mode = parse_trial_answer(candidate, trial)
    return answer, f"reasoning-{mode}", reasoning, final_line, split_mode


def call_model(trial: Trial, config: dict) -> dict:
    if config.get("provider") == "gemini":
        return call_gemini(trial, config)
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": trial.prompt},
        ],
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 256),
    }
    if config.get("reasoning_effort"):
        payload["reasoning_effort"] = config["reasoning_effort"]
    for parameter in (
        "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
        "repetition_penalty", "seed", "thinking_token_budget",
        "thinking_budget_tokens",
    ):
        if parameter in config:
            payload[parameter] = config[parameter]
    if config.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = config["chat_template_kwargs"]
    if config.get("structured_outputs"):
        payload["structured_outputs"] = config["structured_outputs"]
    if config.get("structured_final_answer"):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "kinship_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": list(trial.accepted_answers)}
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    request = urllib.request.Request(
        f"{config['base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=config.get("timeout_seconds", 180)) as response:
        body = json.loads(response.read())
    latency = time.monotonic() - started
    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    finish_reason = body["choices"][0].get("finish_reason")
    truncated = finish_reason == "length"
    separate_reasoning = message.get("reasoning_content") or message.get("reasoning")
    if config.get("structured_final_answer"):
        answer, parse_mode = (None, "truncated") if truncated else parse_structured_trial_answer(content, trial)
        preserved_reasoning = separate_reasoning
        final_content = content
        split_mode = "schema-with-separate-reasoning" if separate_reasoning is not None else "schema-content-only"
    elif config.get("reasoning_enabled") or config.get("final_line_only"):
        answer, parse_mode, preserved_reasoning, final_content, split_mode = (
            parse_reasoning_trial_answer(
                content, trial, separate_reasoning, truncated=truncated
            )
        )
    else:
        answer, parse_mode = parse_trial_answer(content, trial)
        preserved_reasoning = separate_reasoning
        final_content = content
        split_mode = "not-applicable"
    semantic = trial.choices.get(answer) if answer else None
    usage = body.get("usage") or {}
    return {
        "trial_id": trial.trial_id,
        "problem_id": trial.problem_id,
        "condition": trial.condition,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 6),
        "visible_content": content,
        "reasoning_content": preserved_reasoning,
        "final_answer_content": final_content,
        "reasoning_split_mode": split_mode,
        "reasoning_word_count": len((preserved_reasoning or "").split()),
        "final_answer_word_count": len(final_content.split()),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "parsed_letter": answer,
        "parse_mode": parse_mode,
        "parsed_semantic_answer": semantic,
        "correct_letter": trial.correct_letter,
        "correct": answer == trial.correct_letter,
        "usage": usage,
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "response_id": body.get("id"),
    }


def call_gemini(trial: Trial, config: dict) -> dict:
    key = os.environ.get(config.get("api_key_env", "GEMINI_API_KEY"))
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    generation_config = {
        "temperature": config.get("temperature", 0),
        "maxOutputTokens": config.get("max_tokens", 256),
        "thinkingConfig": {"thinkingLevel": config.get("thinking_level", "low")},
    }
    if config.get("structured_output", True):
        generation_config.update({
            "responseMimeType": "text/x.enum",
            "responseSchema": {
                "type": "STRING",
                "enum": list(trial.accepted_answers) if trial.accepted_answers else list("ABCD"),
            },
        })
    payload = {
        "systemInstruction": {"parts": [{"text": config["system_prompt"]}]},
        "contents": [{"role": "user", "parts": [{"text": trial.prompt}]}],
        "generationConfig": generation_config,
    }
    url = (
        f"{config['base_url'].rstrip('/')}/models/{config['model']}:generateContent"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=config.get("timeout_seconds", 180)) as response:
        body = json.loads(response.read())
    latency = time.monotonic() - started
    content = body["candidates"][0]["content"]["parts"][0]["text"]
    answer, parse_mode = parse_trial_answer(content, trial)
    semantic = trial.choices.get(answer) if answer else None
    return {
        "trial_id": trial.trial_id,
        "problem_id": trial.problem_id,
        "condition": trial.condition,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 6),
        "visible_content": content,
        "reasoning_content": None,
        "parsed_letter": answer,
        "parse_mode": parse_mode,
        "parsed_semantic_answer": semantic,
        "correct_letter": trial.correct_letter,
        "correct": answer == trial.correct_letter,
        "usage": body.get("usageMetadata"),
        "response_id": body.get("responseId"),
        "model_version": body.get("modelVersion"),
    }


def run_trials(trials: list[Trial], config: dict, output: Path, limit: int | None = None) -> None:
    fingerprint = configuration_fingerprint(config)
    completed = {
        (row.get("trial_id"), row.get("config_fingerprint"), row.get("trial_fingerprint"))
        for row in read_jsonl(output)
        if not row.get("request_error")
    }
    pending = [
        t for t in trials
        if (t.trial_id, fingerprint, trial_fingerprint(t)) not in completed
    ]
    if limit is not None:
        pending = pending[:limit]
    def execute(trial: Trial) -> dict:
        for attempt in range(9):
            try:
                record = call_model(trial, config)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 8:
                    time.sleep(min(60.0, 2.0 ** attempt) + random.random())
                    continue
                record = {"trial_id":trial.trial_id,"problem_id":trial.problem_id,"condition":trial.condition,"completed_at":datetime.now(timezone.utc).isoformat(),"request_error":f"{type(exc).__name__}: {exc}"}
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                record = {"trial_id":trial.trial_id,"problem_id":trial.problem_id,"condition":trial.condition,"completed_at":datetime.now(timezone.utc).isoformat(),"request_error":f"{type(exc).__name__}: {exc}"}
                break
        record["config_fingerprint"] = fingerprint
        record["trial_fingerprint"] = trial_fingerprint(trial)
        return record

    workers = max(1, int(config.get("concurrency", 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, trial): trial for trial in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            trial = futures[future]
            print(f"[{index}/{len(pending)}] {trial.trial_id}", flush=True)
            append_jsonl(output, future.result())
