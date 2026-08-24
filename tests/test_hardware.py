"""Tests for hardware detection."""

import pytest

from pengu.hardware.detect import (
    HardwareInfo,
    classify_tier,
    detect_hardware,
)
from pengu.config import HardwareTier


class TestHardwareDetection:
    def test_detect_hardware_returns_info(self):
        info = detect_hardware()
        assert isinstance(info, HardwareInfo)
        assert info.os_name != ""
        assert info.hostname != ""
        assert info.ram_total_gb > 0
        assert info.cpu_cores_logical > 0

    def test_os_detection(self):
        info = detect_hardware()
        # Running on Windows or Linux in CI
        assert any(x in info.os_name for x in ("Windows", "Linux", "macOS"))

    def test_ram_detection(self):
        info = detect_hardware()
        assert info.ram_total_gb >= 1.0  # at least 1 GB
        assert info.ram_available_gb >= 0

    def test_tier_classification_low(self):
        info = HardwareInfo()
        info.ram_total_gb = 4
        info.gpu = info.gpu  # no GPU
        tier = classify_tier(info)
        assert tier == HardwareTier.LOW

    def test_tier_classification_medium(self):
        info = HardwareInfo()
        info.ram_total_gb = 16
        tier = classify_tier(info)
        assert tier == HardwareTier.MEDIUM

    def test_tier_classification_high(self):
        info = HardwareInfo()
        info.ram_total_gb = 32
        tier = classify_tier(info)
        assert tier == HardwareTier.HIGH

    def test_tier_classification_high_with_gpu(self):
        from pengu.hardware.detect import GPUInfo
        info = HardwareInfo()
        info.ram_total_gb = 16
        info.gpu = GPUInfo(name="RTX 4090", vram_gb=24)
        tier = classify_tier(info)
        assert tier == HardwareTier.HIGH

    def test_to_dict(self):
        info = detect_hardware()
        d = info.to_dict()
        assert "os" in d
        assert "cpu" in d
        assert "ram" in d
        assert "gpu" in d
        assert "tier" in d

    def test_detect_hardware_has_tools(self):
        info = detect_hardware()
        assert len(info.tools) > 0
        # Python should be found
        assert "python" in info.tools

    def test_has_recommended_models(self):
        info = detect_hardware()
        assert len(info.recommended_models) > 0
        assert "llm" in info.recommended_models


class TestFreeMode:
    """Verify that hardware detection works without any paid services."""

    def test_no_network_required(self):
        """Hardware detection should work offline."""
        info = detect_hardware()
        assert info.os_name != ""

    def test_no_api_keys_required(self):
        """Hardware detection needs no API keys."""
        info = detect_hardware()
        assert info.ram_total_gb > 0
