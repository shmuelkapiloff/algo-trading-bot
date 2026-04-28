# Observability

This folder contains observability assets for Phase 5.2:
- OpenTelemetry instrumentation helpers: `src/monitoring/otel.py`
- Grafana dashboard templates: `observability/grafana/dashboards/`

Import `algotrader_slo.json` into Grafana and map data source to your
Prometheus/OTel pipeline.
