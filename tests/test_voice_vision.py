"""
Tests for Pengu voice and vision providers.
"""
import pytest
from pengu.voice.wake_word import (
    MockWakeWordProvider,
    OpenWakeWordProvider,
    WakeWordEvent,
)
from pengu.voice.stt import MockSTT, FasterWhisperSTT, TranscriptionResult
from pengu.voice.tts import MockTTS, EdgeTTSProvider, SpeechResult
from pengu.vision.provider import MockVisionProvider, LMStudioVisionProvider, VisionResult
from pengu.vision.screen import ScreenCapture, Screenshot


class TestWakeWord:
    def test_mock_wake_word_init(self):
        provider = MockWakeWordProvider()
        assert provider.name == "mock"
        assert provider.phrase == "hello pengu"

    async def test_mock_wake_word_health(self):
        provider = MockWakeWordProvider()
        assert await provider.health_check() is True

    async def test_mock_wake_word_detect(self):
        provider = MockWakeWordProvider()
        provider.simulate_detection(confidence=0.9)
        event = await provider.detect_once()
        assert event is not None
        assert event.confidence == 0.9
        assert event.phrase == "hello pengu"

    async def test_mock_wake_word_no_detection(self):
        provider = MockWakeWordProvider()
        event = await provider.detect_once()
        assert event is None

    def test_openwakeword_init(self):
        provider = OpenWakeWordProvider()
        assert provider.name == "openwakeword"

    def test_wake_word_event(self):
        event = WakeWordEvent(
            timestamp=1000.0,
            confidence=0.85,
            phrase="hello pengu",
            raw_score=0.85,
        )
        assert event.confidence == 0.85


class TestSTT:
    async def test_mock_stt_health(self):
        provider = MockSTT()
        assert await provider.health_check() is True

    async def test_mock_stt_transcribe(self):
        provider = MockSTT()
        provider.set_transcription("Hello Pengu")
        result = await provider.transcribe(b"audio_data")
        assert result is not None
        assert result.text == "Hello Pengu"

    async def test_mock_stt_empty(self):
        provider = MockSTT()
        result = await provider.transcribe(b"audio_data")
        assert result is not None
        assert result.text == ""

    def test_faster_whisper_init(self):
        provider = FasterWhisperSTT(model_size="tiny")
        assert provider.name == "faster-whisper"
        assert provider._model_size == "tiny"


class TestTTS:
    async def test_mock_tts_health(self):
        provider = MockTTS()
        assert await provider.health_check() is True

    async def test_mock_tts_speak(self):
        provider = MockTTS()
        result = await provider.speak("Hello, I am Pengu")
        assert result is not None
        assert result.text_length == len("Hello, I am Pengu")
        assert result.format == "mp3"

    async def test_mock_tts_speak_to_file(self):
        import tempfile
        import os
        provider = MockTTS()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            result = await provider.speak_to_file("Hello", path)
            assert result is not None
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_edge_tts_init(self):
        provider = EdgeTTSProvider()
        assert provider.name == "edge-tts"


class TestVision:
    async def test_mock_vision_health(self):
        provider = MockVisionProvider()
        assert await provider.health_check() is True

    async def test_mock_vision_analyze(self):
        provider = MockVisionProvider()
        result = await provider.analyze_image(b"image_data")
        assert result is not None
        assert "Mock analysis" in result.description

    async def test_mock_vision_screenshot(self):
        provider = MockVisionProvider()
        result = await provider.analyze_screenshot("test.png")
        assert result is not None

    def test_mock_vision_set_result(self):
        provider = MockVisionProvider()
        provider.set_analysis_result("Custom result")
        assert provider._analysis_result == "Custom result"

    def test_lmstudio_vision_init(self):
        provider = LMStudioVisionProvider()
        assert provider.name == "lmstudio-vision"

    def test_vision_result(self):
        result = VisionResult(
            description="A desktop with code",
            confidence=0.9,
            analysis_time_ms=100.0,
            model="test",
            objects=["code", "terminal"],
        )
        assert result.description == "A desktop with code"
        assert len(result.objects) == 2


class TestScreenCapture:
    def test_screen_capture_init(self):
        capture = ScreenCapture()
        assert capture._output_dir.exists()

    async def test_screen_capture_unavailable(self):
        """Test that capture returns None gracefully if PIL not available."""
        capture = ScreenCapture()
        # On a headless server or without PIL, this should return None gracefully
        result = await capture.capture()
        # We can't assert the result because it depends on the environment
        # Just verify it doesn't crash
        assert result is None or isinstance(result, Screenshot)
