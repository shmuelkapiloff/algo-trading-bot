# Event topic name constants — used as routing keys for EventBus.publish()
# Phase 1: in-process asyncio.Queue
# Phase 2: will become Redis Stream names

SIGNAL_GENERATED = "signal.generated"
ORDER_SUBMITTED = "order.submitted"
ORDER_FILLED = "order.filled"
ORDER_REJECTED = "order.rejected"
ORDER_CANCELED = "order.canceled"
ORDER_EXPIRED = "order.expired"
REGIME_UPDATED = "regime.updated"
SYSTEM_STATE_CHANGED = "system.state_changed"  # broker failover, mode transitions
BROKER_HEALTH_CHECK = "broker.health_check"  # periodic health-check result
