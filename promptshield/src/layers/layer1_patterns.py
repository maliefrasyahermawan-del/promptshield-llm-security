import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import PATTERNS_FILE


@dataclass
class LayerResult:
    matched: bool = False
    confidence: float = 0.0
    matched_patterns: list = field(default_factory=list)
    category: str = ""
    reason: str = ""
    latency_ms: float = 0.0


_SEVERITY_CONFIDENCE = {
    "CRITICAL": 0.97,
    "HIGH": 0.82,
    "MEDIUM": 0.71,
    "LOW": 0.55,
}


class PatternMatcher:
    def __init__(self) -> None:
        self._compiled: list[dict] = []
        self._load()

    def _load(self) -> None:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("patterns", []):
            flags = 0
            for flag_name in p.get("flags", "").split("|"):
                flag_name = flag_name.strip()
                if flag_name and hasattr(re, flag_name):
                    flags |= getattr(re, flag_name)
            try:
                compiled = re.compile(p["pattern"], flags)
                self._compiled.append({
                    "id": p["id"],
                    "regex": compiled,
                    "severity": p.get("severity", "MEDIUM"),
                    "category": p.get("category", "unknown"),
                    "description": p.get("description", ""),
                })
            except re.error:
                pass

    def match(self, prompt: str) -> LayerResult:
        start = time.perf_counter()
        hits: list[dict] = []

        for entry in self._compiled:
            if entry["regex"].search(prompt):
                hits.append(entry)

        latency_ms = (time.perf_counter() - start) * 1000

        if not hits:
            return LayerResult(matched=False, confidence=0.0, latency_ms=latency_ms)

        # Severity of worst hit
        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        hits_sorted = sorted(
            hits,
            key=lambda h: severity_order.index(h["severity"]) if h["severity"] in severity_order else 99,
        )
        top = hits_sorted[0]
        base_confidence = _SEVERITY_CONFIDENCE.get(top["severity"], 0.55)

        # Bonus for multiple matches
        confidence = min(1.0, base_confidence + 0.05 * (len(hits) - 1))

        categories = list({h["category"] for h in hits})
        pattern_ids = [h["id"] for h in hits]
        reason = f"Matched {len(hits)} pattern(s): {', '.join(pattern_ids[:5])}"

        return LayerResult(
            matched=True,
            confidence=confidence,
            matched_patterns=pattern_ids,
            category=categories[0] if categories else top["category"],
            reason=reason,
            latency_ms=latency_ms,
        )
