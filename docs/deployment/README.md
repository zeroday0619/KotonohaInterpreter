# Deployment

## Documents

| Document | Scope |
|---|---|
| [Installation and Deployment](installation.md) | Host preparation, source installation, model staging, Jetson and A6000 deployment, and acceptance |
| [Environment Variables](environment.md) | `.env` management, precedence, application variables, and Compose interpolation |
| [Performance Measurement](../performance/measurement.md) | Target measurements and performance acceptance |
| [Operations](../operations/README.md) | Service lifecycle, security, troubleshooting, and observability |

Quick deployment commands:

```bash
bash scripts/manage.sh deploy jetson
bash scripts/manage.sh deploy a6000
bash scripts/manage.sh web a6000
```

Quick uninstall commands preserve models, configuration, logs, and SQLite data:

```bash
bash scripts/manage.sh uninstall jetson
bash scripts/manage.sh uninstall a6000
```

The management entry point delegates to the deployment script. The deployment script
uses `sudo docker` when the current account cannot access the Docker daemon directly. It
forwards only Compose interpolation variables required for deployment, including A6000
GPU assignments. It does not start the interactive orchestrator.
