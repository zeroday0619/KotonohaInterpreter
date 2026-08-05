"""Localize Typer-owned help text through supported command subclasses."""

from __future__ import annotations

from typing import ClassVar

from typer import rich_utils
from typer._click.core import Context, Parameter
from typer._click.formatting import HelpFormatter
from typer.core import TyperCommand, TyperGroup, TyperOption

from kotonoha._i18n import _
from kotonoha._typing import override


def configure_typer_chrome() -> None:
    """Apply the import-time locale to Rich help panel headings."""
    rich_utils.DEPRECATED_STRING = _("(deprecated) ")
    rich_utils.DEFAULT_STRING = _("[default: {}]")
    rich_utils.ENVVAR_STRING = _("[env var: {}]")
    rich_utils.REQUIRED_LONG_STRING = _("[required]")
    rich_utils.ARGUMENTS_PANEL_TITLE = _("Arguments")
    rich_utils.OPTIONS_PANEL_TITLE = _("Options")
    rich_utils.COMMANDS_PANEL_TITLE = _("Commands")
    rich_utils.ERRORS_PANEL_TITLE = _("Error")
    rich_utils.ABORTED_TEXT = _("Aborted.")
    rich_utils.RICH_HELP = _("Try [blue]'{command_path} {help_option}'[/] for help.")


def _localize_option(
    option: TyperOption | None,
    /,
) -> TyperOption | None:
    if option is not None:
        option.help = _("Show this message and exit.")
    return option


def _localize_parameters(
    parameters: list[Parameter],
    /,
) -> list[Parameter]:
    for parameter in parameters:
        if not isinstance(parameter, TyperOption):
            continue
        if "--install-completion" in parameter.opts:
            parameter.help = _("Install completion for the active shell.")
        elif "--show-completion" in parameter.opts:
            parameter.help = _("Print the completion script for the active shell.")
    return parameters


class LocalizedTyperCommand(TyperCommand):
    """Render one Typer command with localized framework help text."""

    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def format_usage(
        self,
        ctx: Context,
        formatter: HelpFormatter,
        /,
    ) -> None:
        formatter.write_usage(
            ctx.command_path,
            " ".join(self.collect_usage_pieces(ctx)),
            prefix=_("Usage: "),
        )

    @override
    def get_help_option(
        self,
        ctx: Context,
        /,
    ) -> TyperOption | None:
        return _localize_option(super().get_help_option(ctx))


class LocalizedTyperGroup(TyperGroup):
    """Render a Typer command group with localized framework help text."""

    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def format_usage(
        self,
        ctx: Context,
        formatter: HelpFormatter,
        /,
    ) -> None:
        formatter.write_usage(
            ctx.command_path,
            " ".join(self.collect_usage_pieces(ctx)),
            prefix=_("Usage: "),
        )

    @override
    def get_help_option(
        self,
        ctx: Context,
        /,
    ) -> TyperOption | None:
        return _localize_option(super().get_help_option(ctx))

    @override
    def get_params(
        self,
        ctx: Context,
        /,
    ) -> list[Parameter]:
        return _localize_parameters(super().get_params(ctx))
