"""Continuous integration workflow contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import yaml

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
WORKFLOW_PATH: Final = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
MANAGEMENT_SCRIPT: Final = PROJECT_ROOT / "scripts" / "manage.sh"
EXPECTED_JOBS: Final = frozenset({"guard", "lint", "ruff", "test"})
# Tokens that identify the runner rather than the quality gate it executes.
RUNNER_TOKENS: Final = frozenset({".", "python", "run", "uv"})
PINNED_ACTION: Final = re.compile(r"^[\w.-]+/[\w.-]+@(v\d+|[0-9a-f]{40})$")


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _workflow_triggers(
    workflow: dict[str, Any],
    /,
) -> dict[str, Any]:
    # PyYAML follows YAML 1.1 and resolves the unquoted `on` key to the boolean True.
    return workflow.get("on", workflow.get(True, {}))


def _run_commands(
    workflow: dict[str, Any],
    /,
) -> str:
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    return "\n".join(str(step.get("run", "")) for step in steps)


def _action_references(
    workflow: dict[str, Any],
    /,
) -> list[str]:
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    return [str(step["uses"]) for step in steps if "uses" in step]


def _local_quality_gates() -> list[str]:
    script = MANAGEMENT_SCRIPT.read_text(encoding="utf-8")
    block = script.split("\n  check)\n", 1)[1].split("\n    ;;", 1)[0]
    gates = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("run_command "):
            continue
        tokens = [
            token
            for token in stripped.removeprefix("run_command ").split()
            if not token.startswith("-") and token not in RUNNER_TOKENS
        ]
        gates.append(" ".join(tokens))
    return gates


def test_continuous_integration_defines_the_four_named_gates() -> None:
    workflow = _workflow()

    assert set(workflow["jobs"]) == set(EXPECTED_JOBS)
    assert all(job["runs-on"] for job in workflow["jobs"].values())


def test_continuous_integration_runs_every_local_quality_gate() -> None:
    gates = _local_quality_gates()
    commands = _run_commands(_workflow())

    assert gates, "scripts/manage.sh check declares no quality gate"
    for gate in gates:
        assert gate in commands, f"`manage.sh check` runs `{gate}` but CI does not"


def test_continuous_integration_guards_the_deployment_contracts() -> None:
    commands = _run_commands(_workflow())

    assert "uv lock --check" in commands
    assert "uv build --wheel" in commands
    assert "--python 3.10" in commands
    assert "bash -n" in commands


def test_continuous_integration_compiles_catalogs_before_checking_them() -> None:
    commands = _run_commands(_workflow())

    compile_command = "uv run --no-sync python scripts/py/i18n.py compile"
    check_command = "uv run --no-sync python scripts/py/i18n.py check"

    assert compile_command in commands
    assert check_command in commands
    assert commands.index(compile_command) < commands.index(check_command)


def test_continuous_integration_installs_only_the_locked_dependency_set() -> None:
    commands = _run_commands(_workflow())

    installs = [line.strip() for line in commands.splitlines() if "uv sync" in line]
    resolutions = [line.strip() for line in commands.splitlines() if "uv run" in line]

    assert installs
    for line in installs:
        assert "--frozen" in line, f"CI may re-resolve the lock: {line}"
    for line in resolutions:
        assert "--frozen" in line or "--no-sync" in line or "--no-project" in line, line


def test_continuous_integration_pins_every_action_and_triggers_on_review() -> None:
    workflow = _workflow()
    triggers = _workflow_triggers(workflow)

    assert "push" in triggers
    assert "pull_request" in triggers
    for reference in _action_references(workflow):
        assert PINNED_ACTION.match(reference), f"unpinned action reference: {reference}"
