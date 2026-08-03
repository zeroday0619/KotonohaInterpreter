"""Structured logging transport and terminal-interface formatting."""

from __future__ import annotations

import json
import logging

from kotonoha.config import load_settings
from kotonoha.logging_setup import (
    TerminalInterfaceLogHandler,
    drain_terminal_interface_logs,
    reset_terminal_interface_logs,
    setup_logging,
)
from kotonoha.tui.log_panel import format_json_log


def test_terminal_logging_keeps_file_json_and_suppresses_raw_console(tmp_path, capsys) -> None:
    log_path = tmp_path / "application.jsonl"
    try:
        logger = setup_logging(
            "INFO",
            log_path,
            console=True,
            service="orchestrator",
            terminal_interface=True,
        )
        logger.info("pipeline.started", mode="remote", roles=["asr", "llm"])

        buffered = drain_terminal_interface_logs()
        assert len(buffered) == 1
        record = json.loads(buffered[0])
        assert record["event"] == "pipeline.started"
        assert record["service"] == "orchestrator"
        assert record["mode"] == "remote"
        assert json.loads(log_path.read_text(encoding="utf-8"))["event"] == "pipeline.started"
        assert capsys.readouterr().err == ""
    finally:
        setup_logging()
        reset_terminal_interface_logs()


def test_terminal_log_buffer_discards_the_oldest_records() -> None:
    reset_terminal_interface_logs()
    handler = TerminalInterfaceLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    for sequence in range(505):
        handler.emit(
            logging.LogRecord(
                "test",
                logging.INFO,
                __file__,
                1,
                json.dumps({"event": "buffer.test", "sequence": sequence}),
                (),
                None,
            )
        )

    records = [json.loads(message) for message in drain_terminal_interface_logs()]
    assert len(records) == 500
    assert records[0]["sequence"] == 5
    assert records[-1]["sequence"] == 504


def test_terminal_log_buffer_drains_bounded_frames_in_order() -> None:
    reset_terminal_interface_logs()
    handler = TerminalInterfaceLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    for sequence in range(3):
        handler.emit(
            logging.LogRecord(
                "test",
                logging.INFO,
                __file__,
                1,
                str(sequence),
                (),
                None,
            )
        )

    assert drain_terminal_interface_logs(maximum=2) == ["0", "1"]
    assert drain_terminal_interface_logs(maximum=2) == ["2"]


def test_json_log_is_rendered_as_a_human_readable_record() -> None:
    rendered = format_json_log(
        json.dumps(
            {
                "service": "tts",
                "event": "tts.loaded",
                "level": "warning",
                "timestamp": "2026-08-04T03:04:05.123456+09:00",
                "backend": "qwen3",
                "fallback": False,
                "placement": {"tts": "remote"},
            }
        )
    )

    assert rendered.plain == (
        '03:04:05 WARNING tts         tts.loaded  backend=qwen3  fallback=false  '
        'placement={"tts":"remote"}'
    )


def test_non_json_log_line_remains_visible() -> None:
    assert format_json_log("plain diagnostic").plain == "plain diagnostic"


def test_tui_logging_is_enabled_by_default() -> None:
    assert load_settings().logging.console
