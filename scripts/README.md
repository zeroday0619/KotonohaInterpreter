# Management Scripts

## Structure

| Path | Purpose |
|---|---|
| `manage.sh` | Confirmed operator entry point, equipment detection, and task routing |
| `deploy.sh` | Jetson and A6000 preparation, deployment, and uninstall implementation |
| `fetch_models.sh` | Offline model snapshot download workflow |
| `py/allocate_gpus.py` | Python entry point for memory-aware A6000 GPU allocation |
| `py/i18n.py` | Python entry point for gettext catalog maintenance |

Use `bash scripts/manage.sh i18n <extract|update|compile|check>` to run catalog
maintenance through the confirmed management entry point.

Place every Python management utility under `scripts/py`. Keep shell entry points under
`scripts` when operators or other shell workflows invoke them directly.

## Safety Contract

- Every `manage.sh` task requires confirmation. Pass `-y` only in controlled automation.
- Target-aware tasks detect the host when `workstation`, `jetson`, or `a6000` is omitted.
- Docker access escalates through `sudo` only when direct daemon access fails.
- Uninstall image removal accepts only repositories with the `kotonohainterpreter-`
  prefix.
- Models, configuration, secrets, logs, and SQLite data remain outside uninstall scope.
