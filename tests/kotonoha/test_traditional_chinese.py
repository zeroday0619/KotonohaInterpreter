"""Traditional Chinese post-processing safety boundaries."""

from __future__ import annotations

from typing import Any

import pytest

from kotonoha.core._zh import TraditionalChineseConverter
from kotonoha.store._db import Store


def test_store_rejects_unbounded_regular_expression_rules(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    store = Store(tmp_path / "rules.db")
    try:
        with pytest.raises(ValueError, match="execution deadline"):
            store.upsert_zh_rules([(r"(a+)+$", "blocked", True, None)])
        assert store.zh_rules() == []
    finally:
        store.close()


def test_converter_skips_a_legacy_regular_expression_rule() -> None:
    converter = TraditionalChineseConverter(
        "missing-opencc-configuration",
        [(r"(a+)+$", "blocked", True)],
    )
    text = "a" * 100 + "!"

    assert converter(text) == text
