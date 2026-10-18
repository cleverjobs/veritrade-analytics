"""Probe the Ollama behaviours the inference design depends on; writes evals/results/ollama_probe.json.
Inputs are synthetic headlines and filler only. Models load one at a time and are unloaded after."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veritrade.config import get_settings
from veritrade.errors import VtError
from veritrade.inference.adapters import (
    AnalyzeResult,
    SummaryResult,
    analyze_stream,
    chat_options,
    classify_signal,
    summarize_stream,
)
from veritrade.inference.confidence import (
    AMBIGUOUS_MASS_MAX,
    first_value_token,
    label_classes,
    label_probabilities,
    score_sign_probabilities,
    sign_classes,
)
from veritrade.inference.incremental import ScoreClosed, SummaryText
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.schemas import ANALYZE_MODELS, LabelOut, OrdinalOut, SummaryOut, ollama_format
from veritrade.inference.types import ChatResult, Endpoint, FieldOrder
from veritrade.models_registry import ModelSpec, allowlist
from veritrade.prompts.registry import PromptSpec
from veritrade.prompts.registry import get as get_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "results" / "ollama_probe.json"

HEADLINES = {
    "bullish": "Quorvane Therapeutics shares jump after its lung cancer drug meets the main goal of a "
    "late-stage trial.",
    "bearish": "Brillstead Motors recalls 410,000 SUVs over brake defects and withdraws its full-year "
    "profit forecast.",
    "neutral": "Ostrava Logistics appoints a new chief financial officer effective January 1.",
    "mixed": "Tamsin Retail beats quarterly sales estimates but warns higher freight costs will squeeze "
    "margins.",
    "bearish_2": "Pellucid Chips cuts 1,800 jobs as orders from smartphone makers fall for a third quarter.",
    "bullish_2": "Marrowfield Energy wins a $3 billion contract to supply gas to a European utility.",
    "neutral_2": "Sundial Insurance will report second-quarter results on August 6.",
    "bearish_3": "Garnet Air shares sink 18% after the airline slashes its annual profit forecast.",
}
FILLER = [
    "Kestrel Grid Systems said quarterly revenue rose 14% to $812 million, ahead of the $790 million "
    "analysts expected.",
    "Adjusted earnings per share came in at $1.37, compared with $1.18 a year earlier.",
    "The company raised its full-year revenue forecast to a range of $3.2 billion to $3.3 billion.",
    "Chief executive Mara Lindqvist said demand from utility customers remained strong through September.",
    "Gross margin widened to 41.2% from 38.9%, helped by lower component costs and higher pricing.",
    "Kestrel ended the quarter with $640 million in cash and no borrowings under its credit facility.",
    "The board approved a $250 million share buyback programme running through the end of next year.",
    "Shares of Kestrel, which trade under the ticker KGSY, rose 6% in extended trading on Tuesday.",
    "The company said it expects to open a second factory in Ohio in the first half of next year.",
    "Analysts at two brokerages said the results confirmed Kestrel's lead in grid-scale battery controllers.",
]
GENERATIVE = [
    "qwen3.5:0.8b-q8_0",
    "qwen3.5:2b-q4_K_M",
    "qwen3.5:4b-q4_K_M",
    "qwen3:4b-instruct-2507-q4_K_M",
    "ministral-3:3b",
    "qwen3.5:9b-q4_K_M",
]
DECISION = ["tev1:0.8b-q8_0", "nimble:9b-q4_K_M", "clef-flash:9b-q8_0"]
LOGPROB_MODELS = {"qwen3.5:0.8b-q8_0", "qwen3.5:4b-q4_K_M"}
ORDER_MODELS = {"qwen3.5:4b-q4_K_M", "qwen3:4b-instruct-2507-q4_K_M", "ministral-3:3b"}
SIZES = {"~300": 300, "~1500": 1500}


def synthetic_article(tokens: int, variant: int = 0) -> str:
    # A distinct opening per run defeats Ollama's prompt-prefix cache, as distinct articles would.
    sentences = [
        f"Update {variant + 1}: Kestrel Grid Systems filed its quarterly report on day {variant + 3}."
    ]
    while sum(len(s) + 1 for s in sentences) < tokens * 4:
        sentences.append(FILLER[(len(sentences) + variant) % len(FILLER)])
    return " ".join(sentences)


def ms(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def p50(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def describe_first_token(chat: ChatResult, key: str) -> dict[str, Any]:
    found = first_value_token(chat.content, chat.logprobs or (), key)
    if found is None:
        return {"found": False}
    chosen, candidates = found
    classes = label_classes if key == "label" else sign_classes
    valid = {rest: p for rest, p in candidates.items() if classes(rest)}
    ambiguous = sum(p for rest, p in valid.items() if rest and len(classes(rest)) > 1)
    top = sorted(candidates.items(), key=lambda kv: -kv[1])[:8]
    return {
        "found": True,
        "chosen": chosen,
        "chosen_classes": sorted(classes(chosen)),
        "chosen_is_argmax": max(valid, key=lambda rest: valid[rest]) == chosen,
        "top": [[rest, round(p, 4), sorted(classes(rest))] for rest, p in top],
        "grammar_valid_candidates": {rest: round(p, 4) for rest, p in valid.items()},
        "grammar_invalid_mass": round(sum(p for rest, p in candidates.items() if not classes(rest)), 4),
        "ambiguous_mass": round(ambiguous, 4),
        "collision": len(classes(chosen)) != 1 or ambiguous > AMBIGUOUS_MASS_MAX * sum(valid.values()),
    }


def summarise_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(samples)}
    keys = sorted({k for s in samples for k, v in s.items() if isinstance(v, int | float)})
    for key in keys:
        out[f"{key}_p50"] = p50([float(s[key]) for s in samples if isinstance(s.get(key), int | float)])
    return out


class Probe:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self.errors: list[str] = []

    async def guard[T](self, label: str, call: Awaitable[T]) -> T | None:
        try:
            return await call
        except Exception as exc:
            # A probe keeps going: one model's failure is a finding, not a reason to lose the rest.
            detail = f"{exc.code}: {exc.message}" if isinstance(exc, VtError) else repr(exc)
            self.errors.append(f"{label}: {detail}")
            print(f"  ! {label}: {detail}", flush=True)
            return None

    async def unload_all(self) -> None:
        for model in await self.client.ps():
            await self.client.unload(model["name"])

    async def logprobs(self, spec: ModelSpec) -> dict[str, Any]:
        """(a) logprobs on non-streaming schema-constrained chat, first value token per schema."""
        prompt = get_prompt("signal", spec.name, kind="generative")
        options = {"num_ctx": spec.num_ctx or 8192, "num_predict": 16}
        out: dict[str, Any] = {}
        for schema, out_type, key in (("label", LabelOut, "label"), ("ordinal", OrdinalOut, "score")):
            rows = []
            for name, headline in HEADLINES.items():
                call = self.client.chat(
                    spec.name,
                    prompt.messages(headline, schema),
                    format=ollama_format(out_type),
                    options=options,
                    logprobs=True,
                    top_logprobs=20,
                )
                result = await self.guard(f"a/{spec.name}/{schema}", call)
                if result is None:
                    continue
                score_probs = score_sign_probabilities if key == "score" else label_probabilities
                probs = score_probs(result.content, result.logprobs)
                rows.append(
                    {
                        "headline": name,
                        "content": result.content,
                        "logprobs_present": result.logprobs is not None,
                        "tokens": [t.token for t in result.logprobs or ()],
                        "first_value_token": describe_first_token(result, key),
                        "class_probabilities": {k: round(v, 4) for k, v in (probs or {}).items()} or None,
                    }
                )
            out[schema] = rows
        return out

    async def stream_logprobs(self, spec: ModelSpec) -> dict[str, Any]:
        """(b) whether streamed chunks carry logprobs."""
        prompt = get_prompt("signal", spec.name, kind="generative")
        final: ChatResult | None = None
        chunks = 0
        async for item in self.client.chat_stream(
            spec.name,
            prompt.messages(HEADLINES["bullish"], "label"),
            format=ollama_format(LabelOut),
            options={"num_ctx": spec.num_ctx or 8192, "num_predict": 16},
            logprobs=True,
            top_logprobs=5,
        ):
            if isinstance(item, ChatResult):
                final = item
            else:
                chunks += 1
        return {
            "content_chunks": chunks,
            "stream_logprobs_present": bool(final and final.logprobs),
            "n_logprob_tokens": len(final.logprobs or ()) if final else 0,
        }

    async def field_order(self, spec: ModelSpec) -> dict[str, Any]:
        """(c) whether generated key order follows the schema's property order."""
        prompt = get_prompt("analyze", spec.name, kind="generative")
        out: dict[str, Any] = {}
        orders: tuple[FieldOrder, ...] = ("sentiment-first", "summary-first")
        for order in orders:
            out_type = ANALYZE_MODELS[order]
            expected = list(out_type.model_fields)
            runs = []
            for name in ("bullish", "bearish", "mixed"):
                text = HEADLINES[name] + " " + synthetic_article(150)
                call = self.client.chat(
                    spec.name,
                    prompt.messages(text, order),
                    format=ollama_format(out_type),
                    options=chat_options(spec, "analyze"),
                )
                result = await self.guard(f"c/{spec.name}/{order}", call)
                if result is None:
                    continue
                try:
                    keys = [k for k, _ in json.loads(result.content, object_pairs_hook=list)]
                except ValueError:
                    keys = []
                runs.append({"headline": name, "keys": keys, "honoured": keys == expected})
            out[order] = {"expected": expected, "runs": runs, "honoured": all(r["honoured"] for r in runs)}
        out["schema_vs_prompt"] = await self.order_conflict(spec)
        return out

    async def order_conflict(self, spec: ModelSpec) -> list[dict[str, Any]]:
        """Prompt asks for one key order, schema for the other: shows whether the grammar decides."""
        prompt = get_prompt("analyze", spec.name, kind="generative")
        pairs: tuple[tuple[FieldOrder, FieldOrder], ...] = (
            ("summary-first", "sentiment-first"),
            ("sentiment-first", "summary-first"),
        )
        rows = []
        for prompt_order, schema_order in pairs:
            out_type = ANALYZE_MODELS[schema_order]
            call = self.client.chat(
                spec.name,
                prompt.messages(HEADLINES["bearish"], prompt_order),
                format=ollama_format(out_type),
                options=chat_options(spec, "analyze"),
            )
            result = await self.guard(f"c/{spec.name}/conflict", call)
            if result is None:
                continue
            keys = [k for k, _ in json.loads(result.content, object_pairs_hook=list)]
            rows.append(
                {
                    "prompt_order": prompt_order,
                    "schema_order": schema_order,
                    "keys": keys,
                    "follows": "schema"
                    if keys == list(out_type.model_fields)
                    else "prompt"
                    if keys == list(ANALYZE_MODELS[prompt_order].model_fields)
                    else "neither",
                }
            )
        return rows

    async def systemone(self, spec: ModelSpec) -> dict[str, Any]:
        """(d) systemone response shape, cold (load) and warm client wall time."""
        prompt = get_prompt("signal", spec.name, kind="decision")
        state = prompt.state(HEADLINES["bullish"])
        question = {"type": "choice", "instructions": prompt.instructions, "criteria": prompt.criteria_map}
        t0 = time.perf_counter()
        raw = await self.client.systemone(spec.name, state, {"sentiment": question})
        cold_ms = (time.perf_counter() - t0) * 1000
        warm: list[float] = []
        choices: dict[str, Any] = {}
        for name, headline in HEADLINES.items():
            for _ in range(3):
                decision = await self.client.decide_sentiment(
                    spec.name, prompt.state(headline), prompt.instructions, prompt.criteria_map
                )
                warm.append(decision.wall_ms)
            choices[name] = {
                "choice": decision.choice,
                "probabilities": {k: round(v, 4) for k, v in decision.probabilities.items()},
                "confidence": round(decision.confidence, 4),
                "input_tokens": decision.input_tokens,
            }
        loaded = [m for m in await self.client.ps() if m["name"] == spec.name]
        return {
            "response_keys": sorted(raw),
            "answer_keys": sorted((raw.get("answers") or {}).get("sentiment") or {}),
            "usage": raw.get("usage"),
            "raw_example": raw,
            "cold_wall_ms": ms(cold_ms),
            "warm_wall_ms_p50": p50(warm),
            "warm_wall_ms_min": ms(min(warm)),
            "warm_wall_ms_max": ms(max(warm)),
            "per_headline": choices,
            "loaded_context_length": loaded[0].get("context_length") if loaded else None,
        }

    async def think_false(self, spec: ModelSpec) -> dict[str, Any]:
        """(e) think=false leaves no thinking text, with and without a JSON-schema format."""
        out: dict[str, Any] = {}
        messages = [{"role": "user", "content": f"Summarise in one sentence: {HEADLINES['mixed']}"}]
        formats: tuple[tuple[str, dict[str, Any] | None], ...] = (
            ("plain", None),
            ("json_schema", ollama_format(SummaryOut)),
        )
        for label, fmt in formats:
            call = self.client.chat_raw(
                spec.name, messages, format=fmt, options={"num_ctx": 4096, "num_predict": 120}
            )
            result = await self.guard(f"e/{spec.name}/{label}", call)
            if result is None:
                continue
            chat, raw = result
            thinking = str((raw.get("message") or {}).get("thinking") or "")
            out[label] = {
                "thinking_chars": len(thinking),
                "think_tag_in_content": "<think>" in chat.content,
                "eval_tokens": chat.timings.eval_tokens,
                "content_head": chat.content[:160],
            }
        cases = [v for v in out.values() if isinstance(v, dict)]
        out["honoured"] = bool(cases) and all(
            c["thinking_chars"] == 0 and not c["think_tag_in_content"] for c in cases
        )
        return out

    async def latency(self, spec: ModelSpec, runs: int) -> dict[str, Any]:
        """(f) warm latency at ~300 and ~1500 input tokens through the serving adapters."""
        out: dict[str, Any] = {}
        for endpoint in spec.endpoints:
            prompt = get_prompt(endpoint, spec.name, kind=spec.kind)
            per_size: dict[str, Any] = {}
            for size_label, tokens in SIZES.items():
                samples: list[dict[str, Any]] = []
                for i in range(runs + 1):
                    label = f"f/{spec.name}/{endpoint}/{size_label}"
                    text = synthetic_article(tokens, variant=i)
                    sample = await self.guard(label, self.one_call(spec, endpoint, prompt, text))
                    if sample is not None and i > 0:
                        samples.append(sample)
                per_size[size_label] = summarise_samples(samples)
            out[endpoint] = per_size
        return out

    async def one_call(
        self, spec: ModelSpec, endpoint: Endpoint, prompt: PromptSpec, text: str
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if endpoint == "signal":
            _verdict, raw = await classify_signal(self.client, spec, prompt, text)
            if isinstance(raw, ChatResult):
                return {
                    "wall_ms": raw.wall_ms,
                    "prompt_tokens": raw.timings.prompt_tokens,
                    "eval_tokens": raw.timings.eval_tokens,
                }
            return {"wall_ms": raw.wall_ms, "prompt_tokens": raw.input_tokens}
        stream = (
            summarize_stream(self.client, spec, prompt, text)
            if endpoint == "summary"
            else analyze_stream(self.client, spec, prompt, text)
        )
        score_ms: float | None = None
        summary_ms: float | None = None
        async for item in stream:
            if isinstance(item, ScoreClosed) and score_ms is None:
                score_ms = (time.perf_counter() - t0) * 1000
            if isinstance(item, SummaryText) and summary_ms is None:
                summary_ms = (time.perf_counter() - t0) * 1000
            if isinstance(item, SummaryResult | AnalyzeResult):
                chat = item.chat
                return {
                    "wall_ms": chat.wall_ms,
                    "ttft_ms": chat.ttft_ms,
                    "time_to_score_ms": score_ms,
                    "time_to_summary_text_ms": summary_ms,
                    "prompt_eval_ms": chat.timings.prompt_eval_ms,
                    "eval_ms": chat.timings.eval_ms,
                    "prompt_tokens": chat.timings.prompt_tokens,
                    "eval_tokens": chat.timings.eval_tokens,
                    "summary_words": len(item.output.summary.split()),
                }
        raise RuntimeError("stream ended without a result")

    async def model(self, spec: ModelSpec, runs: int, sections: dict[str, dict[str, Any]]) -> None:
        name = spec.name
        if spec.kind == "decision":
            if (decision := await self.guard(f"d/{name}", self.systemone(spec))) is not None:
                sections["d_systemone"][name] = decision
        else:
            if (think := await self.guard(f"e/{name}", self.think_false(spec))) is not None:
                sections["e_think_false"][name] = think
            if name in LOGPROB_MODELS:
                if (lp := await self.guard(f"a/{name}", self.logprobs(spec))) is not None:
                    sections["a_logprobs"][name] = lp
                if (sl := await self.guard(f"b/{name}", self.stream_logprobs(spec))) is not None:
                    sections["b_stream_logprobs"][name] = sl
            if name in ORDER_MODELS and (fo := await self.guard(f"c/{name}", self.field_order(spec))):
                sections["c_field_order"][name] = fo
        if runs > 0 and (lat := await self.guard(f"f/{name}", self.latency(spec, runs))) is not None:
            sections["f_latency"][name] = lat


def findings(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for model, schemas in report["a_logprobs"].items():
        for schema, rows in schemas.items():
            firsts = [r["first_value_token"] for r in rows]
            invalid = max((f.get("grammar_invalid_mass", 0) for f in firsts), default=0)
            argmax = sum(1 for f in firsts if f.get("chosen_is_argmax"))
            lines.append(
                f"(a) {model} {schema}: logprobs present={all(r['logprobs_present'] for r in rows)}; "
                f"first value tokens {sorted({f.get('chosen', '?') for f in firsts})}; "
                f"collisions {sum(1 for f in firsts if f.get('collision'))}/{len(rows)}; "
                f"generated token is the reported argmax {argmax}/{len(rows)}; "
                f"max grammar-invalid first-token mass {invalid}"
            )
    for model, info in report["b_stream_logprobs"].items():
        lines.append(f"(b) {model}: streamed chunks carry logprobs={info['stream_logprobs_present']}")
    for model, orders in report["c_field_order"].items():
        verdicts = ", ".join(f"{o}={v['honoured']}" for o, v in orders.items() if isinstance(v, dict))
        conflict = ", ".join(r["follows"] for r in orders.get("schema_vs_prompt", []))
        lines.append(
            f"(c) {model}: schema property order honoured: {verdicts}; "
            f"when prompt and schema disagree the output follows: {conflict or 'n/a'}"
        )
    for model, info in report["d_systemone"].items():
        lines.append(
            f"(d) {model}: keys {info['response_keys']}, answer {info['answer_keys']}; "
            f"cold {info['cold_wall_ms']} ms, warm p50 {info['warm_wall_ms_p50']} ms"
        )
    for model, info in report["e_think_false"].items():
        lines.append(f"(e) {model}: think=false honoured={info['honoured']}")
    for model, endpoints in report["f_latency"].items():
        for endpoint, sizes in endpoints.items():
            parts = []
            for size, s in sizes.items():
                extra = "".join(
                    f", {label} {s[key]}"
                    for key, label in (
                        ("ttft_ms_p50", "ttft"),
                        ("time_to_score_ms_p50", "score"),
                        ("time_to_summary_text_ms_p50", "summary text"),
                    )
                    if s.get(key) is not None
                )
                parts.append(f"{size}: wall {s.get('wall_ms_p50')}{extra} ms")
            lines.append(f"(f) {model} /{endpoint}: " + "; ".join(parts))
    return lines


async def main(runs: int, only: set[str] | None, merge: bool, skip_latency: bool) -> None:
    settings = get_settings()
    specs = allowlist()
    probe = Probe(OllamaClient.from_settings(settings))
    installed = {m["name"] for m in await probe.client.tags()}
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ollama_version": await probe.client.version(),
        "backend": settings.ollama_backend_label,
        "platform": f"{platform.system()} {platform.machine()}",
        "method": {
            "inputs": "synthetic headlines and generated filler; no publisher or dataset text",
            "latency_runs": runs,
            "latency_warmup": 1,
            "decoding": "think=false, temperature 0, seed 0, num_ctx and num_predict from models.toml",
        },
    }
    keys = ("a_logprobs", "b_stream_logprobs", "c_field_order", "d_systemone", "e_think_false", "f_latency")
    sections: dict[str, dict[str, Any]] = {key: {} for key in keys}
    previous: dict[str, Any] = json.loads(OUT.read_text()) if merge and OUT.exists() else {}
    for key in keys:
        sections[key] |= previous.get(key, {})
    for name in [*GENERATIVE, *DECISION]:
        if (only and name not in only) or name not in specs:
            continue
        if name not in installed:
            probe.errors.append(f"{name}: not installed, skipped")
            continue
        print(f"== {name}", flush=True)
        await probe.unload_all()
        await probe.model(specs[name], 0 if skip_latency else runs, sections)
        await probe.unload_all()
    if previous:
        report["merged_models"] = sorted(only or ())
    report |= sections
    report["findings"] = findings(report)
    report["errors"] = probe.errors
    await probe.client.aclose()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("\n".join(report["findings"]))
    print(f"\nwrote {OUT.relative_to(ROOT)}; errors: {len(probe.errors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="warm latency samples per size (after 1 warm-up)")
    parser.add_argument("--only", nargs="*", help="limit to these model tags")
    parser.add_argument("--merge", action="store_true", help="keep other models' results from the last run")
    parser.add_argument("--skip-latency", action="store_true", help="keep (f) from the last run")
    args = parser.parse_args()
    asyncio.run(main(args.runs, set(args.only) if args.only else None, args.merge, args.skip_latency))
