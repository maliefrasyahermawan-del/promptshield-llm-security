import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from src import config
from src.database import init_db, PromptLog, get_stats, get_recent_logs, get_all_logs
from src.metrics import record_request
from src.pipeline import ShieldPipeline

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("promptshield.gateway")

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

_pipeline: ShieldPipeline | None = None

BLOCK_RESPONSE = "Maaf, saya tidak dapat memproses permintaan tersebut."


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    init_db()
    _pipeline = ShieldPipeline()
    logger.info("PromptShield gateway ready on %s:%s", config.HOST, config.PORT)
    yield
    logger.info("PromptShield gateway shutting down")


app = FastAPI(
    title="PromptShield Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------- Request / Response schemas ----------

class ChatRequest(BaseModel):
    prompt: str
    session_id: str = Field(default="")
    target_llm: str = Field(default="")


class ChatResponse(BaseModel):
    response: str
    blocked: bool
    shield_id: str
    latency_ms: float = 0.0


# ---------- Helpers ----------

async def _forward_to_llm(prompt: str) -> str:
    payload = {
        "model": config.TARGET_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(config.TARGET_LLM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", data.get("content", str(data)))
    except Exception as e:
        logger.error("Target LLM error: %s", e)
        return f"Error forwarding to LLM: {e}"


def _save_log(req: ChatRequest, pipeline_result, shield_id: str) -> None:
    try:
        PromptLog.create(
            shield_id=shield_id,
            session_id=req.session_id or "",
            prompt=req.prompt,
            verdict=pipeline_result.verdict,
            triggered_layer=pipeline_result.triggered_layer,
            confidence=pipeline_result.confidence,
            reason=pipeline_result.reason,
            latency_total_ms=pipeline_result.latency_ms.get("total", 0.0),
            latency_layer1_ms=pipeline_result.latency_ms.get("layer1", 0.0),
            latency_layer2_ms=pipeline_result.latency_ms.get("layer2", 0.0),
            latency_layer3_ms=pipeline_result.latency_ms.get("layer3", 0.0),
            similarity_score=pipeline_result.similarity_score,
            matched_patterns=",".join(pipeline_result.matched_patterns),
        )
    except Exception as e:
        logger.error("DB save error: %s", e)


# ---------- API Routes ----------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    pipeline_result = _pipeline.check(req.prompt)
    shield_id = pipeline_result.shield_id
    latency = pipeline_result.latency_ms.get("total", 0.0)

    _save_log(req, pipeline_result, shield_id)
    record_request(pipeline_result.verdict, pipeline_result.triggered_layer, latency)

    if pipeline_result.verdict == "BLOCK":
        logger.info(
            "[BLOCK] sid=%s layer=%d conf=%.3f prompt=%r",
            shield_id, pipeline_result.triggered_layer,
            pipeline_result.confidence, req.prompt[:60],
        )
        return ChatResponse(
            response=BLOCK_RESPONSE,
            blocked=True,
            shield_id=shield_id,
            latency_ms=round(latency, 2),
        )

    logger.info("[ALLOW] sid=%s latency=%.1fms prompt=%r", shield_id, latency, req.prompt[:60])
    llm_response = await _forward_to_llm(req.prompt)
    return ChatResponse(
        response=llm_response,
        blocked=False,
        shield_id=shield_id,
        latency_ms=round(latency, 2),
    )


@app.get("/api/stats")
async def api_stats():
    return get_stats()


@app.get("/api/logs")
async def api_logs(page: int = 1, per_page: int = 50, verdict: str = ""):
    return get_all_logs(page=page, per_page=per_page, verdict_filter=verdict.upper())


@app.get("/api/recent")
async def api_recent(limit: int = 10):
    return get_recent_logs(limit=min(limit, 100))


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": _pipeline is not None}


# ---------- Prometheus ----------

@app.get("/metrics")
async def metrics():
    from starlette.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------- Web Dashboard ----------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats = get_stats()
    recent = get_recent_logs(limit=10)
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "stats": stats, "recent_logs": recent},
    )


@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, page: int = 1, verdict: str = ""):
    data = get_all_logs(page=page, per_page=50, verdict_filter=verdict.upper())
    return templates.TemplateResponse(
        "logs.html",
        {"request": request, **data, "verdict_filter": verdict},
    )


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")
