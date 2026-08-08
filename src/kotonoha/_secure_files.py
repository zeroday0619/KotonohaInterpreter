"""Secure file primitives for local configuration, history, and logs."""

from __future__ import annotations

import errno
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


def reject_symbolic_link(
    path: Path,
    /,
) -> None:
    """Reject a path that currently resolves through a symbolic-link leaf."""
    if path.is_symlink():
        raise OSError(errno.ELOOP, "refusing to follow symbolic link", path)


@contextmanager
def atomic_text_writer(
    path: Path,
    /,
) -> Iterator[TextIO]:
    """Yield an owner-only temporary stream and atomically install it on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    stream: TextIO | None = None
    try:
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None
        temporary_path.replace(path)
        _sync_directory(path.parent)
    finally:
        if stream is not None:
            stream.close()
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    content: str,
    /,
) -> None:
    """Replace a text file atomically without following an existing symlink."""
    with atomic_text_writer(path) as stream:
        stream.write(content)


def _sync_directory(
    path: Path,
    /,
) -> None:
    """Persist the rename when the platform supports directory fsync."""
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        # Some filesystems reject directory fsync even though the rename succeeded.
        return
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def open_append_text(
    path: Path,
    /,
) -> TextIO:
    """Open an owner-only append stream and reject symbolic-link targets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0:
        reject_symbolic_link(path)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | no_follow
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "a", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    return stream


def open_rotating_append_text(
    path: Path,
    maximum_bytes: int,
    backup_count: int,
    /,
) -> TextIO:
    """Open an append stream after rotating a file that reached its size limit."""
    stream = open_append_text(path)
    if os.fstat(stream.fileno()).st_size < maximum_bytes:
        return stream
    stream.close()
    for index in range(backup_count, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if index == backup_count:
            source.unlink(missing_ok=True)
            continue
        if source.exists() or source.is_symlink():
            source.replace(path.with_name(f"{path.name}.{index + 1}"))
    if path.exists() or path.is_symlink():
        path.replace(path.with_name(f"{path.name}.1"))
    return open_append_text(path)
