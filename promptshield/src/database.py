import datetime
from peewee import (
    SqliteDatabase, Model, CharField, FloatField,
    IntegerField, BooleanField, DateTimeField, TextField
)
from src.config import DB_PATH

db = SqliteDatabase(DB_PATH)


class BaseModel(Model):
    class Meta:
        database = db


class PromptLog(BaseModel):
    shield_id = CharField(unique=True, index=True)
    session_id = CharField(default="", index=True)
    prompt = TextField()
    verdict = CharField()                  # ALLOW | BLOCK
    triggered_layer = IntegerField(default=0)
    confidence = FloatField(default=0.0)
    reason = TextField(default="")
    latency_total_ms = FloatField(default=0.0)
    latency_layer1_ms = FloatField(default=0.0)
    latency_layer2_ms = FloatField(default=0.0)
    latency_layer3_ms = FloatField(default=0.0)
    similarity_score = FloatField(default=0.0)
    matched_patterns = TextField(default="")   # comma-separated pattern IDs
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "prompt_logs"


def init_db() -> None:
    db.connect(reuse_if_open=True)
    db.create_tables([PromptLog], safe=True)


def get_stats() -> dict:
    db.connect(reuse_if_open=True)
    total = PromptLog.select().count()
    blocked = PromptLog.select().where(PromptLog.verdict == "BLOCK").count()
    layer1_blocks = PromptLog.select().where(
        (PromptLog.verdict == "BLOCK") & (PromptLog.triggered_layer == 1)
    ).count()
    layer2_blocks = PromptLog.select().where(
        (PromptLog.verdict == "BLOCK") & (PromptLog.triggered_layer == 2)
    ).count()
    layer3_blocks = PromptLog.select().where(
        (PromptLog.verdict == "BLOCK") & (PromptLog.triggered_layer == 3)
    ).count()

    avg_latency = 0.0
    if total > 0:
        query = PromptLog.select(PromptLog.latency_total_ms)
        latencies = [r.latency_total_ms for r in query]
        avg_latency = sum(latencies) / len(latencies)

    # Last 24h hourly counts
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    hourly: dict[int, int] = {h: 0 for h in range(24)}
    for row in PromptLog.select().where(PromptLog.created_at >= cutoff):
        hourly[row.created_at.hour] = hourly.get(row.created_at.hour, 0) + 1

    return {
        "total": total,
        "blocked": blocked,
        "allowed": total - blocked,
        "block_rate": round(blocked / total * 100, 1) if total > 0 else 0.0,
        "avg_latency_ms": round(avg_latency, 1),
        "layer1_blocks": layer1_blocks,
        "layer2_blocks": layer2_blocks,
        "layer3_blocks": layer3_blocks,
        "hourly": hourly,
    }


def get_recent_logs(limit: int = 10) -> list:
    db.connect(reuse_if_open=True)
    rows = (
        PromptLog.select()
        .order_by(PromptLog.created_at.desc())
        .limit(limit)
    )
    result = []
    for r in rows:
        result.append({
            "shield_id": r.shield_id,
            "prompt": r.prompt[:50] + ("..." if len(r.prompt) > 50 else ""),
            "verdict": r.verdict,
            "triggered_layer": r.triggered_layer,
            "confidence": round(r.confidence, 3),
            "latency_ms": round(r.latency_total_ms, 1),
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": r.reason,
        })
    return result


def get_all_logs(page: int = 1, per_page: int = 50, verdict_filter: str = "") -> dict:
    db.connect(reuse_if_open=True)
    query = PromptLog.select().order_by(PromptLog.created_at.desc())
    if verdict_filter in ("ALLOW", "BLOCK"):
        query = query.where(PromptLog.verdict == verdict_filter)
    total = query.count()
    rows = query.paginate(page, per_page)
    logs = []
    for r in rows:
        logs.append({
            "shield_id": r.shield_id,
            "session_id": r.session_id,
            "prompt": r.prompt,
            "verdict": r.verdict,
            "triggered_layer": r.triggered_layer,
            "confidence": round(r.confidence, 3),
            "latency_ms": round(r.latency_total_ms, 1),
            "reason": r.reason,
            "matched_patterns": r.matched_patterns,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return {"logs": logs, "total": total, "page": page, "per_page": per_page}
