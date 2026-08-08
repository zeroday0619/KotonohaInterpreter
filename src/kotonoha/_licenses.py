"""Project and installed dependency license metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from kotonoha._config import REPO_ROOT
from kotonoha._i18n import _

PROJECT_DISTRIBUTION = "kotonoha-interpreter"
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")


@dataclass(frozen=True, slots=True)
class PackageLicense:
    """Describe one installed direct dependency and its declared license."""

    name: str
    version: str
    license_name: str


def project_version() -> str:
    """Return the installed project version."""
    try:
        return distribution(PROJECT_DISTRIBUTION).version
    except PackageNotFoundError:
        return "0.1.0"


def project_license_text() -> str | None:
    """Read the packaged license before falling back to the repository."""
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
    """List direct dependencies present in the runtime environment."""
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
