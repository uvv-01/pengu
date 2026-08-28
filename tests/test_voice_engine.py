"""Tests for the production voice engine and command parser."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from pengu.voice.engine import (
    VoiceConfig,
    VoiceState,
    MicrophoneManager,
    WakeWordDetector,
    CommandRecorder,
    STTEngine,
    TTSEngine,
    VoiceEngine,
)
from pengu.app import CommandParser, ModelDiscovery


# ---------------------------------------------------------------------------
# VoiceConfig
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    def test_default_config(self):
        config = VoiceConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.wake_phrase == "hello pengu"
        assert config.stt_model_size == "tiny"
        assert config.tts_voice == "en-US-GuyNeural"
        assert config.vad_energy_threshold == 15.0
        assert config.command_silence_timeout == 1.5

    def test_custom_config(self):
        config = VoiceConfig(
            sample_rate=44100,
            wake_phrase="hey computer",
            stt_model_size="base",
        )
        assert config.sample_rate == 44100
        assert config.wake_phrase == "hey computer"
        assert config.stt_model_size == "base"


# ---------------------------------------------------------------------------
# MicrophoneManager
# ---------------------------------------------------------------------------

class TestMicrophoneManager:
    def test_init(self):
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        assert not mic.is_active
        assert not mic.is_muted

    def test_energy_calculation(self):
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        silence = np.zeros(1024, dtype=np.int16)
        energy = mic.calculate_energy(silence)
        assert energy == 0.0
        speech = np.random.randint(-1000, 1000, size=3200, dtype=np.int16)
        energy_speech = mic.calculate_energy(speech)
        assert energy_speech > 0.0

    def test_mute_unmute(self):
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        mic.mute()
        assert mic.is_muted
        mic.unmute()
        assert not mic.is_muted

    def test_flush(self):
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        mic._audio_queue.put(np.zeros(100, dtype=np.int16))
        mic._audio_queue.put(np.zeros(100, dtype=np.int16))
        assert not mic._audio_queue.empty()
        mic.flush()
        assert mic._audio_queue.empty()

    def test_get_level(self):
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        level = mic.get_level()
        assert "active" in level
        assert "muted" in level
        assert "avg_rms" in level
        assert "peak" in level

    def test_select_and_start_returns_bool(self):
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        # select_and_start probes all devices and opens a stream
        # On CI without a real mic, this may return False
        result = mic.select_and_start()
        assert isinstance(result, bool)
        if result:
            assert mic.is_active
            assert mic.selection is not None
            mic.stop()


# Regression: check_audio_quality must not raise NameError for MIN_SNR_DB / MIN_SPEECH_RMS
    def test_check_audio_quality_not_ready_path(self):
        """Regression test: check_audio_quality() must execute both quality-gate
        paths without raising NameError for MIN_SNR_DB or MIN_SPEECH_RMS."""
        from pengu.voice.audio_device_manager import (
            AudioQuality, DeviceSelection, MIN_SNR_DB, MIN_SPEECH_RMS,
        )

        config = VoiceConfig()
        mic = MicrophoneManager(config)

        # Simulate a NOT-READY selection (low SNR, no speech)
        bad_sel = DeviceSelection(
            device_index=1,
            device_name="Bad Mic",
            host_api="MME",
            max_channels=2,
            native_sample_rate=44100,
            noise_floor=179.0,
            noise_rms=356.8,
            speech_rms=10.0,       # below MIN_SPEECH_RMS
            speech_peak=200.0,
            rms=10.0,
            peak=200.0,
            snr_db=0.0,            # below MIN_SNR_DB
            clipping_percent=0.0,
            quality=AudioQuality.POOR,
            quality_score=16.0,
            selection_reason="low SNR",
            voice_ready=False,
            speech_detected=False,
        )

        mic._is_active = True
        mic._selection = bad_sel

        ok, report = mic.check_audio_quality()
        assert ok is False
        assert "NOT READY" in report
        assert "MIN_SNR_DB" not in report  # must use value, not name
        assert "6.0" in report              # MIN_SNR_DB value
        assert "100.0" in report            # MIN_SPEECH_RMS value

    def test_check_audio_quality_ready_path(self):
        """Regression test: check_audio_quality() READY path must not raise NameError."""
        from pengu.voice.audio_device_manager import (
            AudioQuality, DeviceSelection,
        )

        config = VoiceConfig()
        mic = MicrophoneManager(config)

        good_sel = DeviceSelection(
            device_index=5,
            device_name="Good Mic",
            host_api="DirectSound",
            max_channels=4,
            native_sample_rate=44100,
            noise_floor=0.5,
            noise_rms=3.6,
            speech_rms=385.0,
            speech_peak=4943.0,
            rms=385.0,
            peak=4943.0,
            snr_db=21.0,
            clipping_percent=0.0,
            quality=AudioQuality.GOOD,
            quality_score=1220.0,
            selection_reason="high SNR, speech detected",
            voice_ready=True,
            speech_detected=True,
        )

        mic._is_active = True
        mic._selection = good_sel

        ok, report = mic.check_audio_quality()
        assert ok is True
        assert "READY" in report

    def test_check_audio_quality_not_active(self):
        """check_audio_quality returns False when mic is inactive."""
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        ok, report = mic.check_audio_quality()
        assert ok is False
        assert "not active" in report.lower()

    def test_check_audio_quality_per_check_gates(self):
        from pengu.voice.audio_device_manager import AudioQuality, DeviceSelection
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        bad_sel = DeviceSelection(
            device_index=1, device_name='Test', host_api='MME',
            max_channels=2, native_sample_rate=44100,
            noise_floor=5.0, noise_rms=5.0,
            speech_rms=10.0, speech_peak=100.0,
            rms=10.0, peak=100.0,
            snr_db=0.0, clipping_percent=0.0,
            quality=AudioQuality.POOR, quality_score=0.0,
            selection_reason='test', voice_ready=False, speech_detected=False,
        )
        mic._is_active = True
        mic._selection = bad_sel
        ok, report = mic.check_audio_quality()
        assert ok is False
        assert '[FAIL] Speech detected' in report
        assert '[FAIL] SNR >=' in report
        assert 'WHAT FAILED' in report
        assert 'Status:       NOT READY' in report

    def test_check_audio_quality_all_ok(self):
        from pengu.voice.audio_device_manager import AudioQuality, DeviceSelection
        config = VoiceConfig()
        mic = MicrophoneManager(config)
        good_sel = DeviceSelection(
            device_index=5, device_name='Good', host_api='DirectSound',
            max_channels=4, native_sample_rate=44100,
            noise_floor=0.5, noise_rms=3.0,
            speech_rms=400.0, speech_peak=5000.0,
            rms=400.0, peak=5000.0,
            snr_db=21.0, clipping_percent=0.0,
            quality=AudioQuality.GOOD, quality_score=1200.0,
            selection_reason='test', voice_ready=True, speech_detected=True,
        )
        mic._is_active = True
        mic._selection = good_sel
        ok, report = mic.check_audio_quality()
        assert ok is True
        assert '[OK] Device opens' in report
        assert '[OK] Speech detected' in report
        assert '[OK] SNR >=' in report
        assert '[OK] Clipping' in report
        assert '[OK] Quality' in report
        assert 'Status:       READY' in report

# ---------------------------------------------------------------------------
# WakeWordDetector
# ---------------------------------------------------------------------------

class TestWakeWordDetector:
    def test_init(self):
        config = VoiceConfig()
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value=None)
        detector = WakeWordDetector(config, stt)
        # Verify internal state is initialized via reset
        detector.reset()
        assert detector._last_wake_time == 0

    def test_no_wake_on_silence(self):
        config = VoiceConfig()
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value=None)
        detector = WakeWordDetector(config, stt)
        silence = np.zeros(1024, dtype=np.int16)
        result = detector.process_chunk_simple(silence, 0.0)
        assert result is None

    def test_wake_detection_with_phrase(self):
        config = VoiceConfig(wake_debounce_seconds=0.1)
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value="hello pengu open vscode")
        detector = WakeWordDetector(config, stt)

        # Simulate speech with realistic timing (need >0.3s total)
        speech = np.ones(1024, dtype=np.int16) * 1000
        for _ in range(5):
            time.sleep(0.1)
            detector.process_chunk_simple(speech, 50.0)

        # Send multiple silence frames to trigger transcription (need >= 3 consecutive)
        time.sleep(0.15)
        silence = np.zeros(1024, dtype=np.int16)
        for _ in range(5):
            result = detector.process_chunk_simple(silence, 0.0)
            if result is not None:
                break
        assert result is not None
        assert "pengu" in result

    def test_no_wake_without_phrase(self):
        config = VoiceConfig(wake_debounce_seconds=0.1)
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value="hello how are you doing today")
        detector = WakeWordDetector(config, stt)

        # Simulate speech then silence with realistic timing
        speech = np.ones(1024, dtype=np.int16) * 1000
        for _ in range(5):
            time.sleep(0.1)
            detector.process_chunk_simple(speech, 50.0)

        time.sleep(0.15)
        silence = np.zeros(1024, dtype=np.int16)
        result = None
        for _ in range(5):
            result = detector.process_chunk_simple(silence, 0.0)
            if result is not None:
                break
        assert result is None

    def test_debounce(self):
        config = VoiceConfig(wake_debounce_seconds=5.0)
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value="hello pengu")
        detector = WakeWordDetector(config, stt)
        detector._last_wake_time = time.time()
        result = detector.process_chunk_simple(np.zeros(1024, dtype=np.int16), 0.0)
        assert result is None

    def test_reset(self):
        config = VoiceConfig()
        stt = MagicMock()
        detector = WakeWordDetector(config, stt)
        detector._last_wake_time = time.time()
        detector.reset()
        assert detector._last_wake_time == 0


# ---------------------------------------------------------------------------
# CommandRecorder
# ---------------------------------------------------------------------------

class TestCommandRecorder:
    def test_init(self):
        config = VoiceConfig()
        recorder = CommandRecorder(config)
        assert not recorder._is_recording
        assert not recorder._has_speech

    def test_record_and_stop(self):
        config = VoiceConfig(command_min_duration=0.1)
        recorder = CommandRecorder(config)
        recorder.start()
        assert recorder._is_recording
        audio = np.random.randint(-1000, 1000, size=3200, dtype=np.int16)
        recorder.add_chunk(audio, 50.0)
        assert recorder._has_speech
        result = recorder.stop()
        assert result is not None
        assert len(result) > 0

    def test_no_audio_returns_none(self):
        config = VoiceConfig()
        recorder = CommandRecorder(config)
        recorder.start()
        result = recorder.stop()
        assert result is None

    def test_silence_timeout(self):
        config = VoiceConfig(command_silence_timeout=0.1, command_min_duration=0.01)
        recorder = CommandRecorder(config)
        recorder.start()
        recorder.add_chunk(np.ones(1024, dtype=np.int16) * 1000, 50.0)
        time.sleep(0.2)
        assert recorder.should_stop()

    def test_max_duration(self):
        config = VoiceConfig(command_max_duration=0.1, command_min_duration=0.01)
        recorder = CommandRecorder(config)
        recorder.start()
        recorder.add_chunk(np.ones(1024, dtype=np.int16) * 1000, 50.0)
        time.sleep(0.15)
        assert recorder.should_stop()


# ---------------------------------------------------------------------------
# STTEngine
# ---------------------------------------------------------------------------

class TestSTTEngine:
    def test_init(self):
        config = VoiceConfig()
        stt = STTEngine(config)
        assert not stt.is_available()

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        config = VoiceConfig()
        stt = STTEngine(config)
        with patch("faster_whisper.WhisperModel") as mock_model:
            result = await stt.initialize()
            assert result is True
            assert stt.is_available()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        config = VoiceConfig()
        stt = STTEngine(config)
        with patch("faster_whisper.WhisperModel", side_effect=ImportError("not installed")):
            result = await stt.initialize()
            assert result is False
            assert not stt.is_available()

    @pytest.mark.asyncio
    async def test_transcribe_empty_when_unavailable(self):
        config = VoiceConfig()
        stt = STTEngine(config)
        audio = np.zeros(16000, dtype=np.int16)
        result = await stt.transcribe(audio)
        assert result is None


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TestTTSEngine:
    def test_init(self):
        config = VoiceConfig()
        tts = TTSEngine(config)
        assert not tts.is_available()
        assert not tts.is_speaking

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        config = VoiceConfig()
        tts = TTSEngine(config)
        with patch("edge_tts.Communicate"):
            result = await tts.initialize()
            assert result is True
            assert tts.is_available()

    @pytest.mark.asyncio
    async def test_speak_returns_false_when_unavailable(self):
        config = VoiceConfig()
        tts = TTSEngine(config)
        result = await tts.speak("hello")
        assert result is False

    def test_cancel(self):
        config = VoiceConfig()
        tts = TTSEngine(config)
        tts.cancel()
        assert tts._cancel_event.is_set()


# ---------------------------------------------------------------------------
# VoiceState
# ---------------------------------------------------------------------------

class TestVoiceState:
    def test_all_states_exist(self):
        expected = [
            "OFFLINE", "STARTING", "STANDBY", "WAKE_DETECTED",
            "ACKNOWLEDGING", "LISTENING", "TRANSCRIBING", "THINKING",
            "EXECUTING", "SPEAKING", "ERROR", "STOPPING",
        ]
        for state_name in expected:
            assert hasattr(VoiceState, state_name)

    def test_state_values(self):
        assert VoiceState.STANDBY.value == "STANDBY"
        assert VoiceState.LISTENING.value == "LISTENING"
        assert VoiceState.SPEAKING.value == "SPEAKING"


# ---------------------------------------------------------------------------
# VoiceEngine (mocked)
# ---------------------------------------------------------------------------

class TestVoiceEngine:
    def test_init(self):
        config = VoiceConfig()
        engine = VoiceEngine(config)
        assert engine.state == VoiceState.OFFLINE
        assert not engine.is_running

    def test_get_status(self):
        config = VoiceConfig()
        engine = VoiceEngine(config)
        status = engine.get_status()
        assert "running" in status
        assert "state" in status
        assert "stt_available" in status
        assert "tts_available" in status
        assert "microphone" in status
        assert "diagnostics" in status

    def test_run_diagnostics(self):
        config = VoiceConfig()
        engine = VoiceEngine(config)
        diag = engine.run_diagnostics()
        assert "microphone" in diag
        assert "stt" in diag
        assert "tts" in diag
        assert "wake_word" in diag
        assert "stats" in diag

    def test_interrupt(self):
        config = VoiceConfig()
        engine = VoiceEngine(config)
        engine.interrupt()
        assert engine._tts._cancel_event.is_set()


# ---------------------------------------------------------------------------
# CommandParser
# ---------------------------------------------------------------------------

class TestCommandParser:
    def setup_method(self):
        self.parser = CommandParser(Path(__file__).parent.parent)

    def test_stop_command(self):
        result = self.parser.parse("stop")
        assert result is not None
        assert result["action"] == "stop"

    def test_cancel_command(self):
        result = self.parser.parse("cancel")
        assert result is not None
        assert result["action"] == "stop"

    def test_help_command(self):
        result = self.parser.parse("what can you do")
        assert result is not None
        assert result["action"] == "help"

    def test_diagnostics_command(self):
        result = self.parser.parse("run diagnostics")
        assert result is not None
        assert result["action"] == "diagnostics"

    def test_open_vscode(self):
        result = self.parser.parse("open vs code")
        assert result is not None
        assert result["action"] == "open_app"

    def test_open_chrome(self):
        result = self.parser.parse("open chrome")
        assert result is not None
        assert result["action"] == "open_app"

    def test_open_explorer(self):
        result = self.parser.parse("open file explorer")
        assert result is not None
        assert result["action"] == "open_app"

    def test_open_terminal(self):
        result = self.parser.parse("open terminal")
        assert result is not None
        assert result["action"] == "open_app"

    def test_open_notepad(self):
        result = self.parser.parse("open notepad")
        assert result is not None
        assert result["action"] == "open_app"

    def test_open_chatgpt(self):
        result = self.parser.parse("open chatgpt")
        assert result is not None
        assert result["action"] == "open_chatgpt"

    def test_open_pengu_folder(self):
        result = self.parser.parse("open pengu")
        assert result is not None
        assert result["action"] == "open_folder"

    def test_open_pengu_in_vscode(self):
        result = self.parser.parse("open pengu in vs code")
        assert result is not None
        assert result["action"] == "open_in_vscode"

    def test_google_search(self):
        result = self.parser.parse("search google for python decorators")
        assert result is not None
        assert result["action"] == "google_search"
        assert "python decorators" in result["speak"]

    def test_search_chatgpt(self):
        result = self.parser.parse("search chatgpt for FastAPI authentication")
        assert result is not None
        assert result["action"] == "open_chatgpt_search"

    def test_open_google(self):
        result = self.parser.parse("open google")
        assert result is not None
        assert result["action"] == "open_url"

    def test_open_youtube(self):
        result = self.parser.parse("open youtube")
        assert result is not None
        assert result["action"] == "open_url"

    def test_git_status(self):
        result = self.parser.parse("git status")
        assert result is not None
        assert result["action"] == "git_result"

    def test_git_log(self):
        result = self.parser.parse("git log")
        assert result is not None
        assert result["action"] == "git_result"

    def test_system_info(self):
        result = self.parser.parse("system information")
        assert result is not None
        assert result["action"] == "system_info"

    def test_what_cpu(self):
        result = self.parser.parse("what cpu do i have")
        assert result is not None
        assert result["action"] == "system_info"

    def test_create_file(self):
        result = self.parser.parse("create a file called test123.py")
        assert result is not None
        assert result["action"] == "file_created"
        try:
            (Path(__file__).parent.parent / "test123.py").unlink()
        except Exception:
            pass

    def test_create_folder(self):
        result = self.parser.parse("create a folder called test_folder_xyz")
        assert result is not None
        assert result["action"] == "folder_created"
        try:
            (Path(__file__).parent.parent / "test_folder_xyz").rmdir()
        except Exception:
            pass

    def test_list_files(self):
        result = self.parser.parse("list files")
        assert result is not None
        assert result["action"] == "list_files"

    def test_close_app(self):
        result = self.parser.parse("close chrome")
        assert result is not None
        assert result["action"] == "close_app"

    def test_unknown_command_returns_none(self):
        result = self.parser.parse("what is the meaning of life")
        assert result is None

    def test_pengu_prefix_stripped(self):
        result = self.parser.parse("pengu, open vs code")
        assert result is not None
        assert result["action"] == "open_app"

    def test_hey_pengu_prefix_stripped(self):
        result = self.parser.parse("hey pengu, open chrome")
        assert result is not None
        assert result["action"] == "open_app"


# ---------------------------------------------------------------------------
# ModelDiscovery
# ---------------------------------------------------------------------------

class TestModelDiscovery:
    def test_init(self):
        discovery = ModelDiscovery()
        assert discovery.active_provider is None
        assert discovery.active_model is None
        assert discovery.available_models == []


# ---------------------------------------------------------------------------
# Integration: STTEngine + WakeWordDetector
# ---------------------------------------------------------------------------

class TestWakeWordIntegration:
    def test_full_wake_sequence(self):
        config = VoiceConfig(wake_debounce_seconds=0.1)
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value="hello pengu open vs code")
        detector = WakeWordDetector(config, stt)

        # Simulate speech with realistic timing
        speech = np.ones(1024, dtype=np.int16) * 1000
        for _ in range(5):
            time.sleep(0.1)
            detector.process_chunk_simple(speech, 50.0)

        # Then multiple silence frames
        time.sleep(0.15)
        silence = np.zeros(1024, dtype=np.int16)
        result = None
        for _ in range(5):
            result = detector.process_chunk_simple(silence, 0.0)
            if result is not None:
                break
        assert result is not None
        assert "pengu" in result

    def test_false_trigger_rejected(self):
        config = VoiceConfig(wake_debounce_seconds=0.1)
        stt = MagicMock()
        stt.transcribe = AsyncMock(return_value="hello how are you doing today")
        detector = WakeWordDetector(config, stt)

        # Simulate speech then silence with realistic timing
        speech = np.ones(1024, dtype=np.int16) * 1000
        for _ in range(5):
            time.sleep(0.1)
            detector.process_chunk_simple(speech, 50.0)

        time.sleep(0.15)
        silence = np.zeros(1024, dtype=np.int16)
        result = None
        for _ in range(5):
            result = detector.process_chunk_simple(silence, 0.0)
            if result is not None:
                break
        assert result is None


# =========================================================================
# Regression tests for AudioDeviceManager, SNR, quality gate
# =========================================================================

class TestAudioDeviceManager:
    def test_no_speech_not_poor(self):
        """A device that opens but has no speech should be NO_SPEECH, not POOR."""
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        q = _classify_quality(
            snr_db=0.0, speech_rms=30.0, clipping_pct=0.0,
            noise_floor=0.5, speech_detected=False, can_open=True,
        )
        assert q == AudioQuality.NO_SPEECH

    def test_no_speech_low_noise_hardware_ok(self):
        from pengu.voice.audio_device_manager import AudioQuality
        assert AudioQuality.NO_SPEECH.is_hardware_ok is True
        assert AudioQuality.NO_SPEECH.is_voice_ready is False

    def test_poor_requires_real_issues(self):
        """POOR should mean actual quality issues, not just no speech."""
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        # Device cannot open -> POOR
        q = _classify_quality(
            snr_db=0.0, speech_rms=0.0, clipping_pct=0.0,
            noise_floor=0.1, speech_detected=False, can_open=False,
        )
        assert q == AudioQuality.POOR

    def test_snr_zero_with_speech_is_poor(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        q = _classify_quality(
            snr_db=0.0, speech_rms=300.0, clipping_pct=0.0,
            noise_floor=5.0, speech_detected=True,
        )
        assert q == AudioQuality.POOR


    def test_snr_calculation_normal(self):
        from pengu.voice.audio_device_manager import _calculate_snr_db
        snr = _calculate_snr_db(speech_rms=500.0, noise_floor=5.0)
        assert abs(snr - 40.0) < 0.1

    def test_snr_calculation_zero_noise(self):
        from pengu.voice.audio_device_manager import _calculate_snr_db
        snr = _calculate_snr_db(speech_rms=500.0, noise_floor=0.0)
        assert snr == 0.0

    def test_snr_calculation_zero_speech(self):
        from pengu.voice.audio_device_manager import _calculate_snr_db
        snr = _calculate_snr_db(speech_rms=0.0, noise_floor=5.0)
        assert snr == 0.0

    def test_snr_calculation_below_min_noise(self):
        from pengu.voice.audio_device_manager import _calculate_snr_db, MIN_NOISE_FLOOR
        snr = _calculate_snr_db(speech_rms=500.0, noise_floor=MIN_NOISE_FLOOR * 0.5)
        assert snr == 0.0

    def test_snr_calculation_equal_signal_noise(self):
        from pengu.voice.audio_device_manager import _calculate_snr_db
        snr = _calculate_snr_db(speech_rms=100.0, noise_floor=100.0)
        assert snr == 0.0

    def test_snr_zero_cannot_become_ready(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        for speech in [100, 300, 500, 1000]:
            q = _classify_quality(snr_db=0.0, speech_rms=speech, clipping_pct=0.0, noise_floor=5.0, speech_detected=True)
            assert q in (AudioQuality.POOR, AudioQuality.UNUSABLE)

    def test_snr_below_min_cannot_become_ready(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality, MIN_SNR_DB
        q = _classify_quality(snr_db=MIN_SNR_DB - 0.1, speech_rms=500.0, clipping_pct=0.0, noise_floor=5.0, speech_detected=True)
        assert q == AudioQuality.POOR

    def test_speech_not_detected_cannot_become_ready(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        # No speech -> NO_SPEECH, NOT voice_ready
        q = _classify_quality(snr_db=30.0, speech_rms=500.0, clipping_pct=0.0, noise_floor=5.0, speech_detected=False)
        assert q == AudioQuality.NO_SPEECH
        assert q.is_voice_ready is False

    def test_excellent_quality_thresholds(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        q = _classify_quality(snr_db=25.0, speech_rms=600.0, clipping_pct=0.0, noise_floor=5.0, speech_detected=True)
        assert q == AudioQuality.EXCELLENT

    def test_good_quality_thresholds(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        q = _classify_quality(snr_db=13.0, speech_rms=250.0, clipping_pct=0.0, noise_floor=5.0, speech_detected=True)
        assert q == AudioQuality.GOOD

    def test_acceptable_quality_thresholds(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality, MIN_SNR_DB
        q = _classify_quality(snr_db=MIN_SNR_DB + 0.1, speech_rms=110.0, clipping_pct=0.0, noise_floor=5.0, speech_detected=True)
        assert q == AudioQuality.ACCEPTABLE

    def test_severe_clipping_unusable(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        q = _classify_quality(snr_db=30.0, speech_rms=500.0, clipping_pct=2.0, noise_floor=5.0, speech_detected=True)
        assert q == AudioQuality.UNUSABLE

    def test_degenerate_signal_unusable(self):
        from pengu.voice.audio_device_manager import _classify_quality, AudioQuality
        # No speech, device opens, low noise -> NO_SPEECH
        q = _classify_quality(snr_db=0.0, speech_rms=0.5, clipping_pct=0.0, noise_floor=0.5, speech_detected=False)
        assert q == AudioQuality.NO_SPEECH
        # speech_detected=True + low signal = UNUSABLE
        q2 = _classify_quality(snr_db=0.0, speech_rms=0.5, clipping_pct=0.0, noise_floor=0.5, speech_detected=True)
        assert q2 == AudioQuality.UNUSABLE

    def test_constants_consistency(self):
        from pengu.voice.audio_device_manager import MIN_SNR_DB, MIN_SPEECH_RMS
        from pengu.voice.engine import MIN_SNR_DB as E_SNR, MIN_SPEECH_RMS as E_RMS
        from pengu.voice.mic_diagnostics import MIN_SNR_DB as D_SNR, MIN_SPEECH_RMS as D_RMS
        assert MIN_SNR_DB == E_SNR == D_SNR == 6.0
        assert MIN_SPEECH_RMS == E_RMS == D_RMS == 100.0
