import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    INJECTION_CORPUS,
    SAFE_CORPUS,
    CLASSIFIER_PATH,
    LAYER2_BLOCK_THRESHOLD,
    LAYER2_SUSPICIOUS_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    matched: bool = False
    confidence: float = 0.0
    matched_patterns: list = field(default_factory=list)
    category: str = "semantic"
    reason: str = ""
    latency_ms: float = 0.0
    similarity_score: float = 0.0
    suspicious: bool = False


class SemanticChecker:
    def __init__(self) -> None:
        self._model = None
        self._injection_embeddings: Optional[np.ndarray] = None
        self._classifier = None
        self._mode = "cosine"
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model loaded (all-MiniLM-L6-v2)")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            return

        # Try loading trained classifier
        if CLASSIFIER_PATH.exists():
            try:
                import joblib
                self._classifier = joblib.load(str(CLASSIFIER_PATH))
                self._mode = "classifier"
                logger.info("Trained classifier loaded from %s", CLASSIFIER_PATH)
                return
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}, falling back to cosine mode")

        # Fallback: load injection corpus and embed
        self._load_cosine_corpus()

    def _load_cosine_corpus(self) -> None:
        if not INJECTION_CORPUS.exists():
            logger.warning("injection_corpus.txt not found — Layer 2 cosine mode disabled")
            return
        try:
            lines = [
                line.strip()
                for line in INJECTION_CORPUS.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines and self._model is not None:
                self._injection_embeddings = self._model.encode(lines, show_progress_bar=False)
                logger.info("Embedded %d injection samples for cosine mode", len(lines))
        except Exception as e:
            logger.error(f"Failed to load injection corpus: {e}")

    def embed(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return self._model.encode([text], show_progress_bar=False)[0]

    def check(self, prompt: str) -> LayerResult:
        start = time.perf_counter()

        if self._model is None:
            latency_ms = (time.perf_counter() - start) * 1000
            return LayerResult(
                reason="Semantic layer unavailable (model not loaded)",
                latency_ms=latency_ms,
            )

        embedding = self.embed(prompt)
        if embedding is None:
            latency_ms = (time.perf_counter() - start) * 1000
            return LayerResult(reason="Embedding failed", latency_ms=latency_ms)

        if self._mode == "classifier" and self._classifier is not None:
            return self._check_classifier(embedding, start)
        elif self._injection_embeddings is not None:
            return self._check_cosine(embedding, start)
        else:
            latency_ms = (time.perf_counter() - start) * 1000
            return LayerResult(
                reason="No injection corpus or classifier available",
                latency_ms=latency_ms,
            )

    def _check_cosine(self, embedding: np.ndarray, start: float) -> LayerResult:
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(
            embedding.reshape(1, -1), self._injection_embeddings
        )[0]
        max_sim = float(np.max(sims))
        latency_ms = (time.perf_counter() - start) * 1000

        blocked = max_sim > LAYER2_BLOCK_THRESHOLD
        suspicious = max_sim > LAYER2_SUSPICIOUS_THRESHOLD and not blocked

        reason = f"Cosine similarity to injection corpus: {max_sim:.3f}"
        return LayerResult(
            matched=blocked,
            confidence=max_sim,
            reason=reason,
            latency_ms=latency_ms,
            similarity_score=max_sim,
            suspicious=suspicious,
        )

    def _check_classifier(self, embedding: np.ndarray, start: float) -> LayerResult:
        proba = self._classifier.predict_proba(embedding.reshape(1, -1))[0]
        injection_prob = float(proba[1])
        latency_ms = (time.perf_counter() - start) * 1000

        blocked = injection_prob > LAYER2_BLOCK_THRESHOLD
        suspicious = injection_prob > LAYER2_SUSPICIOUS_THRESHOLD and not blocked

        reason = f"Classifier injection probability: {injection_prob:.3f}"
        return LayerResult(
            matched=blocked,
            confidence=injection_prob,
            reason=reason,
            latency_ms=latency_ms,
            similarity_score=injection_prob,
            suspicious=suspicious,
        )
