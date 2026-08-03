"""License discovery from source trees and installed distribution metadata."""

from __future__ import annotations

from kotonoha.tui.license_app import installed_direct_dependencies, project_license_text


def test_project_license_is_available_from_the_installed_distribution() -> None:
    license_text = project_license_text()

    assert license_text is not None
    assert license_text.startswith("MIT License")
    assert "Copyright (c) 2026 Euiseo Cha" in license_text


def test_installed_direct_dependencies_include_runtime_components() -> None:
    packages = {package.name.lower(): package for package in installed_direct_dependencies()}

    assert packages["textual"].version
    assert packages["textual"].license_name == "MIT"
    assert packages["pydantic"].license_name == "MIT"
