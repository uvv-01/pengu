"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pengu.config import (
    CostMode,
    PenguConfig,
    load_config,
    get_config,
    HardwareTier,
    AssistantState,
    TaskCategory,
    PermissionLevel,
)


class TestCostMode:
    def test_free_only_is_default(self):
        config = PenguConfig()
        assert config.cost_mode == CostMode.FREE_ONLY

    def test_free_only_blocks_cloud(self):
        config = PenguConfig()
        assert config.is_free_only() is True
        assert config.cloud_enabled() is False

    def test_free_plus_cloud_allows_cloud(self):
        config = PenguConfig(
            cost_mode=CostMode.FREE_PLUS_CLOUD,
            cloud={"gemini_enabled": True, "gemini_api_key": "test-key"},
        )
        assert config.is_free_only() is False

    def test_cost_mode_from_string(self):
        config = PenguConfig(cost_mode="FREE_ONLY")
        assert config.cost_mode == CostMode.FREE_ONLY


class TestPenguConfig:
    def test_defaults(self):
        config = PenguConfig()
        assert config.version == "0.1.0"
        assert config.debug is False
        assert config.api.host == "127.0.0.1"
        assert config.api.port == 8420

    def test_summary(self):
        config = PenguConfig()
        summary = config.summary()
        assert "cost_mode" in summary
        assert "version" in summary
        assert summary["cost_mode"] == "FREE_ONLY"

    def test_ensure_dirs(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        log_dir = tmp_path / "logs"
        config = PenguConfig(data_dir=data_dir, log_dir=log_dir)
        config.ensure_dirs()
        assert data_dir.exists()
        assert log_dir.exists()


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path: Path):
        config_file = tmp_path / "pengu.yaml"
        config_file.write_text(yaml.dump({
            "debug": True,
            "api": {"port": 9999},
        }))
        config = load_config(config_file)
        assert config.debug is True
        assert config.api.port == 9999

    def test_load_defaults_when_no_file(self):
        config = load_config()
        assert config.cost_mode == CostMode.FREE_ONLY


class TestEnums:
    def test_assistant_states(self):
        assert AssistantState.STANDBY.value == "STANDBY"
        assert AssistantState.LISTENING.value == "LISTENING"

    def test_task_categories(self):
        assert TaskCategory.CODING.value == "CODING"

    def test_permission_levels(self):
        assert PermissionLevel.SAFE.value == 0
        assert PermissionLevel.CRITICAL.value == 3

    def test_hardware_tiers(self):
        assert HardwareTier.LOW.value == "LOW"
        assert HardwareTier.HIGH.value == "HIGH"
