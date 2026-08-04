"""Project and installed dependency license information."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

from kotonoha._config import REPO_ROOT
from kotonoha._i18n import _
from kotonoha._typing import override

PROJECT_DISTRIBUTION = "kotonoha-interpreter"
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")


@dataclass(frozen=True, slots=True)
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


def _declared_license(
    metadata: Mapping[str, Any],
    /,
) -> str:
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
    return _("See package metadata")


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
    __slots__: ClassVar[tuple[str, ...]] = ()

    version: str
    license_text: str | None
    dependencies: tuple[PackageLicense, ...]
    title: str
    sub_title: str
    _bindings: BindingsMap

    CSS: ClassVar[str] = """
    Screen { layout: vertical; }
    #license-tabs { height: 1fr; }
    #project-scroll { padding: 1 3; }
    #project-summary { margin-bottom: 1; }
    #license-text { width: 100%; }
    #dependency-notice, #model-notice { margin: 1 2; color: $text-muted; }
    #dependency-table { height: 1fr; margin: 0 2 1 2; }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("p", "project", ""),
        ("d", "dependencies", ""),
        ("q", "back", ""),
    ]

    @override
    def __init__(
        self,
        /,
    ) -> None:
        super().__init__()
        self.version = "0.1.0"
        self.license_text = None
        self.dependencies = ()
        self._bindings = BindingsMap(
            [
                Binding("p", "project", _("Project")),
                Binding("d", "dependencies", _("Dependencies")),
                Binding("q", "back", _("Back")),
            ]
        )

    @override
    def compose(
        self,
        /,
    ) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="project", id="license-tabs"):
            with TabPane(_("Project license"), id="project"):
                with VerticalScroll(id="project-scroll"):
                    yield Static(self._project_summary(), id="project-summary")
                    yield Static(
                        self.license_text
                        or _("The project license text is unavailable in this installation."),
                        markup=False,
                        id="license-text",
                    )
            with TabPane(_("Installed dependencies"), id="dependencies"):
                yield Static(
                    _(
                        "License identifiers are read from installed package metadata. "
                        "Retain any license files required by each distributed package."
                    ),
                    id="dependency-notice",
                )
                yield DataTable(zebra_stripes=True, cursor_type="row", id="dependency-table")
                yield Static(
                    _(
                        "Downloaded model artifacts are not covered by the project MIT "
                        "license. Review and retain the license files supplied with each model."
                    ),
                    id="model-notice",
                )
        yield Footer()

    def _project_summary(
        self,
        /,
    ) -> Text:
        summary = Text()
        summary.append(_("Product:") + " ", style="dim")
        summary.append("Kotonoha Interpreter\n", style="bold")
        summary.append(_("Version:") + " ", style="dim")
        summary.append(self.version + "\n")
        summary.append(_("License:") + " ", style="dim")
        summary.append("MIT")
        return summary

    async def on_mount(
        self,
        /,
    ) -> None:
        self.title = _("License information")
        self.sub_title = _("Project and installed dependencies")
        self.version, self.license_text, self.dependencies = await asyncio.gather(
            asyncio.to_thread(project_version),
            asyncio.to_thread(project_license_text),
            asyncio.to_thread(installed_direct_dependencies),
        )
        self.query_one("#project-summary", Static).update(self._project_summary())
        self.query_one("#license-text", Static).update(
            self.license_text
            or _("The project license text is unavailable in this installation.")
        )
        table = self.query_one("#dependency-table", DataTable)
        table.add_columns(
            _("Package"),
            _("Version"),
            _("Declared license"),
        )
        for package in self.dependencies:
            table.add_row(package.name, package.version, package.license_name)
        if not self.dependencies:
            table.add_row(_("No direct dependencies detected"), "", "")

    def action_project(
        self,
        /,
    ) -> None:
        self.query_one("#license-tabs", TabbedContent).active = "project"

    def action_dependencies(
        self,
        /,
    ) -> None:
        self.query_one("#license-tabs", TabbedContent).active = "dependencies"

    @override
    def action_back(
        self,
        /,
    ) -> None:
        self.exit(None)
