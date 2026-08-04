"""Adapters for frameworks that invoke callbacks with keyword arguments."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Final, TypeVar, cast

CallableType = TypeVar("CallableType", bound=Callable[..., Any])

_MISSING_POSITIONAL_ARGUMENT: Final = object()


def keyword_compatible(
    function: CallableType,
    /,
) -> CallableType:
    """Preserve a positional-only declaration for keyword-invoking frameworks.

    Typer and FastAPI inspect the wrapped signature, then invoke callbacks with
    keyword arguments. The wrapper transfers positional-only values back to
    their declared positions before dispatching the original function.
    """
    positional_names = tuple(
        parameter.name
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    )

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def asynchronous_wrapper(
            positional_argument: object = _MISSING_POSITIONAL_ARGUMENT,
            /,
            *arguments: Any,
            **keyword_arguments: Any,
        ) -> Any:
            leading = _leading_arguments(
                positional_names,
                positional_argument,
                keyword_arguments,
            )
            return await function(*leading, *arguments, **keyword_arguments)

        return cast(CallableType, asynchronous_wrapper)

    @wraps(function)
    def synchronous_wrapper(
        positional_argument: object = _MISSING_POSITIONAL_ARGUMENT,
        /,
        *arguments: Any,
        **keyword_arguments: Any,
    ) -> Any:
        leading = _leading_arguments(
            positional_names,
            positional_argument,
            keyword_arguments,
        )
        return function(*leading, *arguments, **keyword_arguments)

    return cast(CallableType, synchronous_wrapper)


def _leading_arguments(
    positional_names: tuple[str, ...],
    /,
    positional_argument: object,
    keyword_arguments: dict[str, Any],
) -> tuple[Any, ...]:
    leading: list[Any] = []
    if positional_argument is not _MISSING_POSITIONAL_ARGUMENT:
        leading.append(positional_argument)
    for name in positional_names[len(leading) :]:
        if name in keyword_arguments:
            leading.append(keyword_arguments.pop(name))
    return tuple(leading)
