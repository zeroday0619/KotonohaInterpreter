# Deployment

## Documents

| Document | Scope |
|---|---|
| [Installation and Deployment](installation.md) | Host preparation, source installation, model staging, Jetson and A6000 deployment, and acceptance |
| [Performance Measurement](../performance/measurement.md) | Target measurements and performance acceptance |
| [Operations](../operations/README.md) | Service lifecycle, security, troubleshooting, and observability |

Quick deployment commands:

```bash
bash scripts/deploy.sh jetson
bash scripts/deploy.sh a6000
```

Quick uninstall commands preserve models, configuration, logs, and SQLite data:

```bash
bash scripts/deploy.sh uninstall jetson
bash scripts/deploy.sh uninstall a6000
```

The deployment script uses `sudo docker` when the current account cannot access the
Docker daemon directly. It does not start the interactive orchestrator.
