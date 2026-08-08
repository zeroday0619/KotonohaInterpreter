"""UI-neutral catalog and validation for operator subprocess commands."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from kotonoha._i18n import N_, _, current_locale

OPERATIONS = (
    "replay",
    "devices",
    "serve",
    "doctor",
    "netcheck",
    "glossary_import",
    "glossary_list",
    "completion_show",
    "completion_install",
)

OPERATION_FIELDS: dict[str, tuple[str, ...]] = {
    "replay": ("wav", "replay-seconds"),
    "devices": (),
    "serve": ("service", "host", "port"),
    "doctor": (),
    "netcheck": ("samples", "netcheck-seconds"),
    "glossary_import": ("glossary-path",),
    "glossary_list": (),
    "completion_show": (),
    "completion_install": (),
}

OPERATION_LABELS: dict[str, str] = {
    "completion_install": N_("Install shell completion"),
    "completion_show": N_("Show shell completion"),
    "devices": N_("List audio devices"),
    "doctor": N_("Run environment diagnostics"),
    "glossary_import": N_("Import a glossary"),
    "glossary_list": N_("List glossary entries"),
    "netcheck": N_("Measure the external link"),
    "replay": N_("Replay a WAV file"),
    "serve": N_("Start a model service"),
}

OPERATION_DESCRIPTIONS: dict[str, str] = {
    "completion_install": N_("Install completion for the active shell."),
    "completion_show": N_("Print the completion script for the active shell."),
    "devices": N_("Print available audio devices and system defaults."),
    "doctor": N_("Inspect dependencies, placement, models, and service health."),
    "glossary_import": N_("Load glossary terms and Chinese conversion rules."),
    "glossary_list": N_("Print every term stored in the local glossary."),
    "netcheck": N_("Measure remote service latency and audio upload throughput."),
    "replay": N_("Run the full pipeline from a 16-bit PCM WAV file."),
    "serve": N_("Start one Python ASR service."),
}


class ToolInputError(ValueError):
    """Report invalid command input before a child process starts."""

    __slots__: ClassVar[tuple[str, ...]] = ()


def _positive_integer(
    value: str,
    /,
    field: str,
    *,
    maximum: int | None = None,
) -> str:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ToolInputError(_("{field} must be a positive integer.", field=field)) from error
    if parsed <= 0:
        raise ToolInputError(_("{field} must be a positive integer.", field=field))
    if maximum is not None and parsed > maximum:
        raise ToolInputError(_("{field} must not exceed {maximum}.", field=field, maximum=maximum))
    return str(parsed)


def _positive_number(
    value: str,
    /,
    field: str,
) -> str:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ToolInputError(_("{field} must be a positive number.", field=field)) from error
    if parsed <= 0:
        raise ToolInputError(_("{field} must be a positive number.", field=field))
    return str(parsed)


def _existing_file(
    value: str,
    /,
    field: str,
) -> str:
    path = Path(value).expanduser()
    if not value.strip() or not path.is_file():
        raise ToolInputError(_("{field} must reference an existing file.", field=field))
    return str(path)


def build_tool_command(
    operation: str,
    /,
    values: Mapping[str, str],
    config_path: Path | None = None,
) -> list[str]:
    """Build a validated CLI invocation without involving a shell."""
    if operation not in OPERATIONS:
        raise ToolInputError(_("Select a valid operation."))
    command = [sys.executable, "-m", "kotonoha._cli"]
    if config_path is not None:
        command.extend(("--config", str(config_path)))
    command.extend(("--lang", current_locale()))
    if operation == "replay":
        command.extend(
            (
                "replay",
                _existing_file(values.get("wav", ""), _("WAV file")),
                "--seconds",
                _positive_number(values.get("replay-seconds", ""), _("Duration in seconds")),
            )
        )
    elif operation == "devices":
        command.append("devices")
    elif operation == "serve":
        service = values.get("service", "")
        if service not in {"asr", "verify"}:
            raise ToolInputError(_("Select a valid service."))
        host = values.get("host", "").strip()
        if not host:
            raise ToolInputError(_("{field} is required.", field=_("Bind address")))
        command.extend(("serve", service, "--host", host))
        port = values.get("port", "").strip()
        if port:
            command.extend(("--port", _positive_integer(port, _("Port"), maximum=65535)))
    elif operation == "doctor":
        command.append("doctor")
    elif operation == "netcheck":
        command.extend(
            (
                "netcheck",
                "--samples",
                _positive_integer(values.get("samples", ""), _("Measurements per role")),
                "--seconds",
                _positive_number(values.get("netcheck-seconds", ""), _("Duration in seconds")),
            )
        )
    elif operation == "glossary_import":
        command.extend(
            (
                "glossary",
                "import",
                _existing_file(values.get("glossary-path", ""), _("Glossary YAML file")),
            )
        )
    elif operation == "glossary_list":
        command.extend(("glossary", "list"))
    elif operation == "completion_show":
        command.append("--show-completion")
    else:
        command.append("--install-completion")
    return command
