"""Compile the translation catalogs during the build.

`.po` files are committed; `.mo` files are not. They are generated here, so an
installed copy carries compiled catalogs while the repository keeps a single source
of truth and no artifact that can fall out of step with it.

The hook runs for every hatchling target, editable installs included, which is how
the Jetson containers get their catalogs: they install with
`uv pip install --system --no-deps -e .`, and build isolation supplies Babel.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LOCALE_DIR = Path("src") / "kotonoha" / "locale"
DOMAIN = "kotonoha"


class LocaleBuildHook(BuildHookInterface):
    PLUGIN_NAME = "kotonoha-locale"

    def initialize(self, version: str, build_data: dict) -> None:
        from babel.messages.mofile import write_mo
        from babel.messages.pofile import read_po

        root = Path(self.root)
        compiled: list[str] = []

        for source in sorted((root / LOCALE_DIR).glob(f"*/LC_MESSAGES/{DOMAIN}.po")):
            with source.open("rb") as handle:
                catalog = read_po(handle, domain=DOMAIN)
            target = source.with_suffix(".mo")
            with target.open("wb") as handle:
                # Fuzzy entries are excluded, which is what gettext tooling does:
                # an unreviewed translation should show English, not guesswork.
                write_mo(handle, catalog, use_fuzzy=False)
            compiled.append(str(target.relative_to(root)))

        if not compiled:
            self.app.display_warning("no .po catalogs found; the interface will be English only")
            return

        # Editable installs read from the source tree, so writing the files is
        # already enough. A wheel needs them listed, because .mo is gitignored and
        # hatchling's default file selection follows version control.
        force_include = build_data.setdefault("force_include", {})
        for relative in compiled:
            force_include[str(root / relative)] = relative.replace("src/", "", 1)

        self.app.display_info(f"compiled {len(compiled)} translation catalogs")
