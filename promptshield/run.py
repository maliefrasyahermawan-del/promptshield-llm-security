"""
PromptShield entry point.
Usage: python run.py
"""
import sys
import uvicorn
from src.config import HOST, PORT, LOG_LEVEL

if __name__ == "__main__":
    print("=" * 56)
    print("  🛡  PromptShield Gateway")
    print("=" * 56)
    print(f"  URL      : http://{HOST if HOST != '0.0.0.0' else 'localhost'}:{PORT}")
    print(f"  Dashboard: http://localhost:{PORT}/dashboard")
    print(f"  API docs : http://localhost:{PORT}/docs")
    print(f"  Metrics  : http://localhost:{PORT}/metrics")
    print("=" * 56)
    uvicorn.run(
        "src.gateway:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        reload=False,
    )
