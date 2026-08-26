"""
Voice pipeline — wake word, STT, TTS for Pengu.

Provides clean interfaces for:
  - WakeWordProvider: detect "Hello Pengu"
  - SpeechToTextProvider: transcribe audio to text
  - TextToSpeechProvider: speak responses

All providers are optional. Text mode always works without voice.
"""

from __future__ import annotations

from pengu.voice.wake_word import WakeWordProvider, get_wake_word_provider
from pengu.voice.stt import SpeechToTextProvider, get_stt_provider
from pengu.voice.tts import TextToSpeechProvider, get_tts_provider

__all__ = [
    "WakeWordProvider",
    "SpeechToTextProvider",
    "TextToSpeechProvider",
    "get_wake_word_provider",
    "get_stt_provider",
    "get_tts_provider",
]
