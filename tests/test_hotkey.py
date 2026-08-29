"""Tests for GlobalHotkey — Windows global hotkey support."""

import pytest
from unittest.mock import MagicMock, patch
from pengu.hotkey import GlobalHotkey, get_hotkey, MOD_CONTROL, MOD_ALT


class TestGlobalHotkey:
    """Test GlobalHotkey functionality."""

    def test_initial_state(self):
        hotkey = GlobalHotkey()
        assert hotkey.is_registered is False

    @patch("pengu.hotkey.user32", None)
    def test_register_not_windows(self):
        hotkey = GlobalHotkey()
        callback = MagicMock()
        result = hotkey.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)
        assert result is False

    @patch("pengu.hotkey.user32")
    def test_register_success(self, mock_user32):
        mock_user32.RegisterHotKey.return_value = 1  # non-zero = success
        hotkey = GlobalHotkey()
        callback = MagicMock()
        result = hotkey.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)
        assert result is True
        assert hotkey.is_registered is True
        mock_user32.RegisterHotKey.assert_called_once()

    @patch("pengu.hotkey.user32")
    def test_register_failure(self, mock_user32):
        mock_user32.RegisterHotKey.return_value = 0  # zero = failure
        hotkey = GlobalHotkey()
        callback = MagicMock()
        result = hotkey.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)
        assert result is False
        assert hotkey.is_registered is False

    @patch("pengu.hotkey.user32")
    def test_unregister(self, mock_user32):
        mock_user32.RegisterHotKey.return_value = 1
        hotkey = GlobalHotkey()
        callback = MagicMock()
        hotkey.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)
        assert hotkey.is_registered is True

        hotkey.unregister()
        assert hotkey.is_registered is False
        mock_user32.UnregisterHotKey.assert_called_once()

    @patch("pengu.hotkey.user32")
    def test_register_default(self, mock_user32):
        mock_user32.RegisterHotKey.return_value = 1
        hotkey = GlobalHotkey()
        callback = MagicMock()
        result = hotkey.register_default(callback)
        assert result is True

    @patch("pengu.hotkey.user32")
    def test_stop(self, mock_user32):
        mock_user32.RegisterHotKey.return_value = 1
        hotkey = GlobalHotkey()
        callback = MagicMock()
        hotkey.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)
        hotkey.start()
        assert hotkey._running is True
        hotkey.stop()
        assert hotkey._running is False


class TestHotkeySingleton:
    """Test the singleton pattern."""

    def test_get_hotkey_returns_same_instance(self):
        h1 = get_hotkey()
        h2 = get_hotkey()
        assert h1 is h2
