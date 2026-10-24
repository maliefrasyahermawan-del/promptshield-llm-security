import logging
import time
import uuid
from dataclasses import dataclass, field

from src.layers.layer1_patterns import PatternMatcher
from src.layers.layer2_semantic import SemanticChecker
from src.layers.layer3_llm_judge import LLMJudge
from src.config import (
    LAYER1_BLOCK_THRESHOLD,
    LAYER1_SUSPICIOUS_THRESHOLD,
    LAYER2_BLOCK_THRESHOLD,
    LAYER2_SUSPICIOUS_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    verdict: str = "ALLOW"
    confidence: float = 0.0
    triggered_layer: int = 0
    reason: str = ""
    latency_ms: dict = field(default_factory=dict)
    matched_patterns: list = field(default_factory=list)
    similarity_score: float = 0.0
    shield_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class ShieldPipeline:
    def __init__(self) -> None:
        logger.info("Initializing ShieldPipeline...")
        self._layer1 = PatternMatcher()
        logger.info("Layer 1 (PatternMatcher) ready")
        self._layer2 = SemanticChecker()
        logger.info("Layer 2 (SemanticChecker) ready")
        self._layer3 = LLMJudge()
        logger.info("Layer 3 (LLMJudge) ready")

    def check(self, prompt: str) -> PipelineResult:
        result = PipelineResult()
        latency: dict[str, float] = {}
        suspicious = False
        pipeline_start = time.perf_counter()

        # --- Layer 1: Pattern Matching ---
        l1 = self._layer1.match(prompt)
        latency["layer1"] = l1.latency_ms

        if l1.matched:
            result.matched_patterns = l1.matched_patterns
            if l1.confidence >= LAYER1_BLOCK_THRESHOLD:
                result.verdict = "BLOCK"
                result.confidence = l1.confidence
                result.triggered_layer = 1
                result.reason = l1.reason
                latency["layer2"] = 0.0
                latency["layer3"] = 0.0
                latency["total"] = (time.perf_counter() - pipeline_start) * 1000
                result.latency_ms = latency
                return result
            elif l1.confidence >= LAYER1_SUSPICIOUS_THRESHOLD:
                suspicious = True
                logger.debug("Layer 1 suspicious (%.3f): %s", l1.confidence, l1.reason)

        # --- Layer 2: Semantic Check ---
        l2 = self._layer2.check(prompt)
        latency["layer2"] = l2.latency_ms
        result.similarity_score = l2.similarity_score

        if l2.matched and l2.confidence >= LAYER2_BLOCK_THRESHOLD:
            result.verdict = "BLOCK"
            result.confidence = l2.confidence
            result.triggered_layer = 2
            result.reason = l2.reason
            latency["layer3"] = 0.0
            latency["total"] = (time.perf_counter() - pipeline_start) * 1000
            result.latency_ms = latency
            return result
        elif l2.confidence >= LAYER2_SUSPICIOUS_THRESHOLD or getattr(l2, "suspicious", False):
            suspicious = True
            logger.debug("Layer 2 suspicious (%.3f): %s", l2.confidence, l2.reason)

        # --- Layer 3: LLM Judge (only if suspicious) ---
        if suspicious:
            l3 = self._layer3.judge(prompt)
            latency["layer3"] = l3.latency_ms

            if not l3.judge_available:
                logger.warning("LLM Judge unavailable, running 2-layer mode")
            elif l3.matched:
                result.verdict = "BLOCK"
                result.confidence = l3.confidence
                result.triggered_layer = 3
                result.reason = l3.reason
                latency["total"] = (time.perf_counter() - pipeline_start) * 1000
                result.latency_ms = latency
                return result
            else:
                logger.debug("Layer 3 cleared: %s", l3.reason)
        else:
            latency["layer3"] = 0.0

        latency["total"] = (time.perf_counter() - pipeline_start) * 1000
        result.latency_ms = latency
        result.verdict = "ALLOW"
        result.triggered_layer = 0
        result.reason = "Passed all layers"
        return result
