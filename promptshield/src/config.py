import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT: int = _int("OLLAMA_TIMEOUT_SECONDS", 10)

TARGET_LLM_URL: str = os.getenv("TARGET_LLM_URL", "http://localhost:11434/api/generate")
TARGET_LLM_MODEL: str = os.getenv("TARGET_LLM_MODEL", "llama3.2:3b")

LAYER1_BLOCK_THRESHOLD: float = _float("LAYER1_BLOCK_THRESHOLD", 0.95)
LAYER1_SUSPICIOUS_THRESHOLD: float = _float("LAYER1_SUSPICIOUS_THRESHOLD", 0.70)
LAYER2_BLOCK_THRESHOLD: float = _float("LAYER2_BLOCK_THRESHOLD", 0.85)
LAYER2_SUSPICIOUS_THRESHOLD: float = _float("LAYER2_SUSPICIOUS_THRESHOLD", 0.65)

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = _int("PORT", 8000)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

DB_PATH: str = os.getenv("DB_PATH", str(BASE_DIR / "promptshield.db"))

DATA_DIR: Path = BASE_DIR / "data"
PATTERNS_FILE: Path = DATA_DIR / "injection_patterns.json"
INJECTION_CORPUS: Path = DATA_DIR / "injection_corpus.txt"
SAFE_CORPUS: Path = DATA_DIR / "safe_corpus.txt"
CLASSIFIER_PATH: Path = BASE_DIR / "src" / "layers" / "classifier.joblib"
