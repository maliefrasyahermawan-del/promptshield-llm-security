"""
Train Layer 2 LogisticRegression classifier from injection + safe corpus.
Usage: python scripts/train_classifier.py
Output: src/layers/classifier.joblib
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from src.config import INJECTION_CORPUS, SAFE_CORPUS, CLASSIFIER_PATH


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  Loaded {len(lines)} lines from {path.name}")
    return lines


def main() -> None:
    print("=" * 60)
    print("  PromptShield — Layer 2 Classifier Training")
    print("=" * 60)

    print("\n[1/5] Loading corpus...")
    injection_lines = load_lines(INJECTION_CORPUS)
    safe_lines = load_lines(SAFE_CORPUS)

    print("\n[2/5] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    print("      (First run downloads ~80MB model — please wait)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("      Model loaded.")

    print("\n[3/5] Embedding corpus...")
    print("      Embedding injection samples...")
    inj_embeddings = model.encode(injection_lines, show_progress_bar=True, batch_size=32)
    print("      Embedding safe samples...")
    safe_embeddings = model.encode(safe_lines, show_progress_bar=True, batch_size=32)

    X = np.vstack([inj_embeddings, safe_embeddings])
    y = np.array([1] * len(injection_lines) + [0] * len(safe_lines))
    print(f"\n      Dataset: {len(injection_lines)} injection + {len(safe_lines)} safe = {len(y)} total samples")
    print(f"      Embedding shape: {X.shape}")

    print("\n[4/5] Training LogisticRegression with 5-fold cross-validation...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
    print(f"\n  Cross-validation F1 scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train final model on full dataset
    clf.fit(X, y)

    # Evaluate on training set (for reference)
    y_pred = clf.predict(X)
    print("\n[5/5] Training set evaluation:")
    print(classification_report(y, y_pred, target_names=["safe", "injection"], digits=3))

    cm = confusion_matrix(y, y_pred)
    print("  Confusion Matrix (safe=0, injection=1):")
    print(f"  {'':>12}  Pred: safe  Pred: injection")
    print(f"  True: safe       {cm[0][0]:>8}  {cm[0][1]:>14}")
    print(f"  True: injection  {cm[1][0]:>8}  {cm[1][1]:>14}")

    # Save model
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, str(CLASSIFIER_PATH))
    print(f"\n  Classifier saved → {CLASSIFIER_PATH}")
    print("\nDone! Restart the gateway to use the trained classifier.")
    print("=" * 60)


if __name__ == "__main__":
    main()
