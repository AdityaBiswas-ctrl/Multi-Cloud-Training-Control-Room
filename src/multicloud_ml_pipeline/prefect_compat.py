"""Compatibility shims so the project can be imported without Prefect installed."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def _identity_decorator(func: F | None = None, **_: Any) -> Callable[[F], F] | F:
    def decorator(inner: F) -> F:
        @wraps(inner)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return inner(*args, **kwargs)

        return cast(F, wrapped)

    if func is not None:
        return decorator(func)
    return decorator


class _FallbackLogger:
    def info(self, message: str, *args: Any) -> None:
        print(message % args if args else message)

    def warning(self, message: str, *args: Any) -> None:
        print(message % args if args else message)

    def error(self, message: str, *args: Any) -> None:
        print(message % args if args else message)


try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
except ImportError:  # pragma: no cover - exercised only when Prefect is absent.
    flow = _identity_decorator
    task = _identity_decorator

    def get_run_logger() -> _FallbackLogger:
        return _FallbackLogger()
