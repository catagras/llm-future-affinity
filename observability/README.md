# Local audit backend

The runner uses Grafana's single-container OTEL-LGTM stack for Tempo traces, Loki audit logs, and Grafana queries.

```powershell
docker compose -f observability/compose.yaml up -d
docker compose -f observability/compose.yaml down
```

The persisted `otel-data/` directory is retained by `down`. The application never manages Docker itself.

