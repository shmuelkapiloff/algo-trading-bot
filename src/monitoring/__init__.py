"""Monitoring layer public API."""

from .logger import configure_logging
from .alerts import AlertDispatcher, AlertLevel

# TcaMonitor and CanaryProbe available via direct import:
#   from src.monitoring.tca import TcaMonitor
#   from src.monitoring.canary_probe import CanaryProbe
# (not re-exported here to avoid pulling in alpaca-py at import time)

__all__ = [
    "configure_logging",
    "AlertDispatcher",
    "AlertLevel",
]
