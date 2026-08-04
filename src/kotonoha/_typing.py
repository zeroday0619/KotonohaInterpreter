"""Typing compatibility helpers for the Python 3.10 target."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

CallableType = TypeVar("CallableType", bound=Callable[..., Any])

if TYPE_CHECKING:
    from typing_extensions import override as override
else:

    def override(
        function: CallableType,
        /,
    ) -> CallableType:
        """Mark a method as an override on runtimes before Python 3.12."""
        function.__override__ = True
        return function
