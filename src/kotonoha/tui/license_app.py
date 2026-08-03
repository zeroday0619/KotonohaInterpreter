"""Project and installed dependency license information."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

from ..config import REPO_ROOT
from ..i18n import t

PROJECT_DISTRIBUTION = "kotonoha-interpreter"
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class PackageLicense:
    """One installed direct dependency and its declared license metadata."""

    name: str
    version: str
    license_name: str


def project_version() -> str:
    try:
        return distribution(PROJECT_DISTRIBUTION).version
    except PackageNotFoundError:
        return "0.1.0"


def project_license_text() -> str | None:
    """Read the packaged license first so installed wheels do not need the repository."""
    try:
        package_distribution = distribution(PROJECT_DISTRIBUTION)
        for file in package_distribution.files or ():
            if file.name.upper() == "LICENSE":
                license_path = Path(file.locate())
                if license_path.is_file():
                    return license_path.read_text(encoding="utf-8")
        metadata_license = package_distribution.metadata.get("License")
        if metadata_license and "Permission is hereby granted" in metadata_license:
            return metadata_license
    except (PackageNotFoundError, OSError, UnicodeError):
        pass

    repository_license = REPO_ROOT / "LICENSE"
    try:
        return repository_license.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _declared_license(metadata: Mapping[str, Any]) -> str:
    expression = metadata.get("License-Expression")
    if expression:
        return str(expression)

    license_value = str(metadata.get("License") or "").strip()
    if license_value and "\n" not in license_value and len(license_value) <= 80:
        return license_value

    classifiers = metadata.get_all("Classifier") if hasattr(metadata, "get_all") else []
    for classifier in classifiers or ():
        prefix = "License :: OSI Approved :: "
        if classifier.startswith(prefix):
            return classifier.removeprefix(prefix)
    return t("license.unknown")


def installed_direct_dependencies() -> tuple[PackageLicense, ...]:
    """List direct dependencies present in the current runtime environment."""
    try:
        requirements = distribution(PROJECT_DISTRIBUTION).requires or ()
    except PackageNotFoundError:
        return ()

    packages: dict[str, PackageLicense] = {}
    for requirement in requirements:
        match = REQUIREMENT_NAME.match(requirement)
        if match is None:
            continue
        requirement_name = match.group(1)
        try:
            dependency = distribution(requirement_name)
        except PackageNotFoundError:
            continue
        canonical_name = dependency.metadata.get("Name") or requirement_name
        packages[canonical_name.lower()] = PackageLicense(
            name=canonical_name,
            version=dependency.version,
            license_name=_declared_license(dependency.metadata),
        )
    return tuple(sorted(packages.values(), key=lambda package: package.name.lower()))


class LicenseApp(App[None]):
    """Display legal information without requiring a browser or network access."""

    CSS = """
    Screen { layout: vertical; }
    #license-tabs { height: 1fr; }
    #project-scroll { padding: 1 3; }
    #project-summary { margin-bottom: 1; }
    #license-text { width: 100%; }
    #dependency-notice, #model-notice { margin: 1 2; color: $text-muted; }
    #dependency-table { height: 1fr; margin: 0 2 1 2; }
    """

    BINDINGS = [
        ("p", "project", ""),
        ("d", "dependencies", ""),
        ("q", "back", ""),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.version = project_version()
        self.license_text = project_license_text()
        self.dependencies = installed_direct_dependencies()
        self._bindings = BindingsMap(
            [
                Binding("p", "project", t("license.key.project")),
                Binding("d", "dependencies", t("license.key.dependencies")),
                Binding("q", "back", t("license.key.back")),
            ]
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="project", id="license-tabs"):
            with TabPane(t("license.tab.project"), id="project"):
                with VerticalScroll(id="project-scroll"):
                    yield Static(self._project_summary(), id="project-summary")
                    yield Static(
                        self.license_text or t("license.unavailable"),
                        markup=False,
                        id="license-text",
                    )
            with TabPane(t("license.tab.dependencies"), id="dependencies"):
                yield Static(t("license.dependencies.notice"), id="dependency-notice")
                yield DataTable(zebra_stripes=True, cursor_type="row", id="dependency-table")
                yield Static(t("license.models.notice"), id="model-notice")
        yield Footer()

    def _project_summary(self) -> Text:
        summary = Text()
        summary.append(t("license.project.name") + " ", style="dim")
        summary.append("Kotonoha Interpreter\n", style="bold")
        summary.append(t("license.project.version") + " ", style="dim")
        summary.append(self.version + "\n")
        summary.append(t("license.project.type") + " ", style="dim")
        summary.append("MIT")
        return summary

    def on_mount(self) -> None:
        self.title = t("license.title")
        self.sub_title = t("license.subtitle")
        table = self.query_one("#dependency-table", DataTable)
        table.add_columns(
            t("license.column.package"),
            t("license.column.version"),
            t("license.column.license"),
        )
        for package in self.dependencies:
            table.add_row(package.name, package.version, package.license_name)
        if not self.dependencies:
            table.add_row(t("license.none"), "", "")

    def action_project(self) -> None:
        self.query_one("#license-tabs", TabbedContent).active = "project"

    def action_dependencies(self) -> None:
        self.query_one("#license-tabs", TabbedContent).active = "dependencies"

    def action_back(self) -> None:
        self.exit(None)
