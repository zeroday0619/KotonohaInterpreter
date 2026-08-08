"""Hardware-profile selection and direct service-client contracts."""

from __future__ import annotations

from pathlib import Path

from kotonoha._config import ROLES, load_settings
from kotonoha._config_store import apply_changes, reset_changes
from kotonoha._settings_database import read_snapshot, seed_defaults, write_overrides
from kotonoha.clients._build import build_service_group


def test_profile_selection_applies_target_specific_defaults() -> None:
    assert apply_changes({"accelerator.profile": "nvidia.rtx.a6000"}).written

    settings = load_settings()

    assert settings.accelerator.profile == "nvidia.rtx.a6000"
    assert settings.asr.vllm_realtime_architecture == "voxtral"
    assert settings.asr_verify.device == "cuda"
    assert settings.llm.profiles["translategemma"].directory == "translategemma-12b-it"


def test_reset_preserves_the_selected_profile_and_removes_overrides() -> None:
    assert apply_changes(
        {
            "accelerator.profile": "nvidia.rtx.a6000",
            "session.fixed_target": "ja",
        }
    ).written

    result = reset_changes()
    settings = load_settings()

    assert result.changed == ["session.fixed_target"]
    assert settings.accelerator.profile == "nvidia.rtx.a6000"
    assert settings.session.fixed_target == "en"


def test_packaged_defaults_refresh_without_replacing_operator_state(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    database = tmp_path / "settings.db"
    profiles = {"vendor.family.model": {"runtime": {"limit": 2}}}
    seed_defaults(database, {"runtime": {"enabled": True}}, profiles, "vendor.family.model")
    write_overrides(
        database,
        {"runtime": {"limit": 9}},
        "vendor.family.model",
    )

    seed_defaults(
        database,
        {"runtime": {"enabled": False}},
        {"vendor.family.model": {"runtime": {"limit": 3}}},
        "vendor.family.model",
    )
    snapshot = read_snapshot(database, "vendor.family.model")

    assert snapshot.defaults == {"runtime": {"enabled": False, "limit": 3}}
    assert snapshot.overrides == {"runtime": {"limit": 9}}
    assert snapshot.effective == {"runtime": {"enabled": False, "limit": 9}}


def test_legacy_yaml_is_imported_once(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("session:\n  fixed_target: ja\n", encoding="utf-8")

    assert load_settings(legacy).session.fixed_target == "ja"
    legacy.write_text("session:\n  fixed_target: ko\n", encoding="utf-8")

    assert load_settings(legacy).session.fixed_target == "ja"


def test_reset_rejects_a_cross_field_invalid_default_combination() -> None:
    assert apply_changes(
        {
            "frontend.vad.threshold": 0.9,
            "frontend.vad.neg_threshold": 0.8,
        }
    ).written

    result = reset_changes(["frontend.vad.threshold"])

    assert result.written is False
    assert result.error is not None
    settings = load_settings()
    assert settings.frontend.vad.threshold == 0.9
    assert settings.frontend.vad.neg_threshold == 0.8


async def test_service_group_uses_one_endpoint_per_role() -> None:
    settings = load_settings()
    group = build_service_group(settings)

    try:
        assert {client.role for client in group.all()} == set(ROLES)
        assert {
            client.role: client.active.base_url for client in group.all()
        } == {
            role: settings.service_url(role).rstrip("/") for role in ROLES
        }
        assert all(client.degraded is False for client in group.all())
    finally:
        await group.aclose()
