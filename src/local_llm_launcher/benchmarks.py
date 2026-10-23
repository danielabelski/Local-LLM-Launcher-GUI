"""Summarize measurements already emitted by inference engines."""
from __future__ import annotations

import math
import re
from typing import Iterable

_NUMBER = r"\d+(?:\.\d+)?"
_TIMING = re.compile(rf"\b(?P<phase>prompt eval|eval) time\s*=\s*(?P<ms>{_NUMBER}) ms\s*/\s*(?P<tokens>\d+) (?:tokens|runs)\b")
_THROUGHPUT = re.compile(rf"Avg (?P<phase>prompt|generation) throughput:\s*(?P<tps>{_NUMBER}) tokens/s\b")


def summarize(lines: Iterable[str], engine: str) -> dict:
    totals = {phase: {"tokens": 0, "ms": 0.0, "rates": 0.0, "samples": 0}
              for phase in ("prefill", "generation")}
    llama = engine == "llamacpp"
    for line in lines:
        for match in (_TIMING if llama else _THROUGHPUT).finditer(line):
            phase = "prefill" if match["phase"] in ("prompt eval", "prompt") else "generation"
            total = totals[phase]
            if llama:
                ms, tokens = float(match["ms"]), int(match["tokens"])
                if ms <= 0 or tokens <= 0 or not math.isfinite(ms):
                    continue
                total["ms"] += ms
                total["tokens"] += tokens
            else:
                rate = float(match["tps"])
                if not math.isfinite(rate):
                    continue
                total["rates"] += rate
            total["samples"] += 1

    result = {"basis": "tokens / measured time" if llama else "mean of logged intervals"}
    for phase, total in totals.items():
        count = total["samples"]
        result[f"{phase}_samples"] = count
        result[f"{phase}_tps"] = (
            total["tokens"] * 1000 / total["ms"] if llama else total["rates"] / count
        ) if count else None
        result[f"{phase}_ms"] = total["ms"] if llama and count else None
        result["prompt_tokens" if phase == "prefill" else "generated_tokens"] = (
            total["tokens"] if llama and count else None)
    return result
