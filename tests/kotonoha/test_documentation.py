"""Documentation category and local-link contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final
from urllib.parse import unquote

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOCUMENTATION_ROOT: Final[Path] = PROJECT_ROOT / "docs"
CATEGORY_INDEXES: Final[tuple[Path, ...]] = tuple(
    DOCUMENTATION_ROOT / category / "README.md"
    for category in (
        "architecture",
        "deployment",
        "development",
        "operations",
        "performance",
        "planning",
        "user-guide",
    )
)
LINK_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _documentation_paths() -> tuple[Path, ...]:
    category_documents = tuple(sorted(DOCUMENTATION_ROOT.rglob("*.md")))
    return (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "CLAUDE.md",
        PROJECT_ROOT / "spikes" / "README.md",
        *category_documents,
    )


def _local_link_targets(
    source: Path,
    /,
) -> tuple[Path, ...]:
    targets: list[Path] = []
    for raw_target in LINK_PATTERN.findall(source.read_text(encoding="utf-8")):
        target = raw_target.partition("#")[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        targets.append((source.parent / unquote(target)).resolve())
    return tuple(targets)


def test_documentation_categories_exist() -> None:
    assert (DOCUMENTATION_ROOT / "README.md").is_file()
    assert all(path.is_file() for path in CATEGORY_INDEXES)


def test_implementation_plan_defines_all_phase_gates() -> None:
    plan = (DOCUMENTATION_ROOT / "planning" / "README.md").read_text(encoding="utf-8")

    for phase in range(6):
        assert f"## Phase {phase}:" in plan
    assert "explicit approval" in plan
    assert "Source implementation alone does not satisfy a phase gate" in plan


def test_local_documentation_links_resolve() -> None:
    missing_links: list[str] = []
    for source in _documentation_paths():
        for target in _local_link_targets(source):
            if not target.exists():
                missing_links.append(f"{source.relative_to(PROJECT_ROOT)} -> {target}")

    assert not missing_links, "\n".join(missing_links)
