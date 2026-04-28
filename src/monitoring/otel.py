"""OpenTelemetry helpers with graceful no-op fallback.

If opentelemetry is not installed, tracing calls become no-ops.
"""

from __future__ import annotations

from contextlib import contextmanager


def is_enabled() -> bool:
    try:
        import opentelemetry  # noqa: F401
        return True
    except Exception:
        return False


@contextmanager
def span(name: str, **attrs):
    """Create an OTel span if available, else no-op context manager."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("algotrader")
        with tracer.start_as_current_span(name) as s:
            for k, v in attrs.items():
                s.set_attribute(k, v)
            yield s
    except Exception:
        yield None


def add_event(name: str, **attrs) -> None:
    """Add event to current OTel span if available."""
    try:
        from opentelemetry import trace

        span_obj = trace.get_current_span()
        if span_obj is not None:
            span_obj.add_event(name, attributes=attrs)
    except Exception:
        return
