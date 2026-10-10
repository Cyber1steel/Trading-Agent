"""Versioned live evaluation dataset and no-credential safety gate tests."""

import json
from pydantic import SecretStr

from app.core.config import Settings
from scripts import evaluate_reasoning_provider
from tests.test_reasoning_evaluation import build_phase3b_golden_scenarios


def test_live_dataset_is_versioned_and_matches_offline_phase3b_scenarios():
    dataset = json.loads(evaluate_reasoning_provider.DATASET_PATH.read_text(encoding="utf-8"))
    fixture_ids = {item.scenario_id for item in build_phase3b_golden_scenarios()}
    dataset_ids = {item["scenario_id"] for item in dataset["scenarios"]}
    assert dataset["dataset_id"] == "reasoning-adversarial-golden"
    assert dataset["dataset_version"] == 1
    assert len(dataset_ids) == 13
    assert dataset_ids == fixture_ids


def test_live_runner_skips_without_credentials_even_when_cost_flag_is_set(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_reasoning_provider,
        "get_settings",
        lambda: Settings(_env_file=None, groq_api_key=None, openai_api_key=SecretStr("openai-only")),
    )
    assert evaluate_reasoning_provider.main(["--provider", "groq", "--confirm-live-cost"]) == 0
    output = capsys.readouterr().out
    assert "SKIPPED" in output
    assert "GROQ_API_KEY" in output
    assert "openai-only" not in output


def test_live_runner_requires_explicit_cost_confirmation_with_credentials(monkeypatch):
    monkeypatch.setattr(
        evaluate_reasoning_provider,
        "get_settings",
        lambda: Settings(_env_file=None, groq_api_key=SecretStr("test-only")),
    )
    try:
        evaluate_reasoning_provider.main(["--provider", "groq"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("live evaluation must require cost confirmation")
