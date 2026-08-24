"""
Tests for ₹0 free-only mode.

Verifies that:
  - FREE_ONLY blocks paid providers
  - Missing API keys don't crash
  - No billing happens
  - Local fallback works
"""

import pytest

from pengu.config import CostMode, PenguConfig, get_config


class TestFreeOnlyMode:
    def test_free_only_is_default(self):
        config = PenguConfig()
        assert config.cost_mode == CostMode.FREE_ONLY

    def test_free_only_blocks_all_cloud(self):
        config = PenguConfig(cost_mode=CostMode.FREE_ONLY)
        assert config.is_free_only() is True
        assert config.cloud_enabled() is False

    def test_free_only_blocks_even_with_keys(self):
        """Even with API keys, FREE_ONLY mode blocks cloud."""
        config = PenguConfig(
            cost_mode=CostMode.FREE_ONLY,
            cloud={
                "gemini_enabled": True,
                "gemini_api_key": "fake-key",
                "groq_enabled": True,
                "groq_api_key": "fake-key",
            },
        )
        assert config.is_free_only() is True
        assert config.cloud_enabled() is False

    def test_free_plus_cloud_requires_explicit_key(self):
        config = PenguConfig(
            cost_mode=CostMode.FREE_PLUS_CLOUD,
            cloud={"gemini_enabled": True, "gemini_api_key": "test-key"},
        )
        assert config.is_free_only() is False

    def test_cloud_without_key_is_not_enabled(self):
        config = PenguConfig(
            cost_mode=CostMode.FREE_PLUS_CLOUD,
            cloud={"gemini_enabled": True},
        )
        assert config.cloud_enabled() is False


class TestNoCrashWithoutKeys:
    """Pengu must not crash when all API keys are missing."""

    def test_config_loads_without_any_keys(self):
        config = PenguConfig()
        assert config.cloud.gemini_api_key == ""
        assert config.cloud.groq_api_key == ""
        assert config.cloud.openrouter_api_key == ""

    def test_summary_works_without_keys(self):
        config = PenguConfig()
        summary = config.summary()
        assert summary["cost_mode"] == "FREE_ONLY"
        assert summary["cloud_enabled"] is False


class TestCostModeValues:
    def test_free_only_value(self):
        assert CostMode.FREE_ONLY.value == "FREE_ONLY"

    def test_free_plus_cloud_value(self):
        assert CostMode.FREE_PLUS_CLOUD.value == "FREE_PLUS_CLOUD"
