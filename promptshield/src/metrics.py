from prometheus_client import Counter, Histogram, Gauge, REGISTRY

requests_total = Counter(
    "promptshield_requests_total",
    "Total number of requests processed",
    ["verdict"],
)

blocked_total = Counter(
    "promptshield_blocked_total",
    "Total number of blocked requests",
)

layer_triggered_count = Counter(
    "promptshield_layer_triggered_total",
    "Number of times each layer triggered a BLOCK",
    ["layer"],
)

latency_histogram = Histogram(
    "promptshield_request_latency_ms",
    "Request processing latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

active_sessions = Gauge(
    "promptshield_active_sessions",
    "Approximate number of distinct sessions seen in last hour",
)


def record_request(verdict: str, triggered_layer: int, latency_ms: float) -> None:
    requests_total.labels(verdict=verdict).inc()
    latency_histogram.observe(latency_ms)
    if verdict == "BLOCK":
        blocked_total.inc()
        if triggered_layer > 0:
            layer_triggered_count.labels(layer=str(triggered_layer)).inc()
