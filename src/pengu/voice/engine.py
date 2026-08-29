"""
Pengu Voice Engine — production voice assistant pipeline.

Architecture:
  STANDBY -> VAD detects speech -> transcribe -> check "hello pengu"
  -> if wake phrase: ACKNOWLEDGE -> LISTEN command -> TRANSCRIBE -> EXECUTE -> SPEAK -> STANDBY

All microphone device selection goes through AudioDeviceManager.
There is NO duplicated device-ranking logic in this file.
"""

from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from pengu.logging import get_logger
from pengu.voice.audio_device_manager import (
    AudioDeviceManager,
    DeviceSelection,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    AudioQuality,
    MIN_SNR_DB,
    MIN_SPEECH_RMS,
    _downmix_to_mono,
    _resample,
)

logger = get_logger("pengu.voice.engine")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class VoiceState(str, Enum):
    OFFLINE = "OFFLINE"
    STARTING = "STARTING"
    STANDBY = "STANDBY"
    WAKE_DETECTED = "WAKE_DETECTED"
    ACKNOWLEDGING = "ACKNOWLEDGING"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
    STOPPING = "STOPPING"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VoiceConfig:
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = TARGET_CHANNELS
    device_index: Optional[int] = None
    vad_energy_threshold: float = 15.0
    vad_silence_timeout: float = 1.0
    vad_max_speech_duration: float = 5.0
    vad_adaptive: bool = True
    wake_phrase: str = "hello pengu"
    wake_debounce_seconds: float = 3.0
    command_silence_timeout: float = 1.5
    command_min_duration: float = 0.3
    command_max_duration: float = 20.0
    stt_model_size: str = "tiny"
    stt_language: str = "en"
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"
    barge_in_energy_threshold: float = 30.0


# ---------------------------------------------------------------------------
# MicrophoneManager — delegates to AudioDeviceManager
# ---------------------------------------------------------------------------

class MicrophoneManager:
    """
    Streaming microphone capture.

    Device selection is handled entirely by AudioDeviceManager.
    This class only manages the persistent stream and audio queue.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._stream = None
        self._is_active = False
        self._is_muted = False
        self._lock = threading.Lock()
        self._device_info: dict[str, Any] = {}
        self._rms_history: list[float] = []
        self._peak_level: float = 0.0
        self._noise_floor: float = 0.5
        self._calibrated = False
        self._selection: Optional[DeviceSelection] = None

    def select_and_start(self) -> bool:
        """
        Select the best device using AudioDeviceManager, then open a persistent stream.
        This is the ONLY place device selection happens.
        """
        # Use AudioDeviceManager — the single source of truth
        configured = self._config.device_index
        env_device = os.environ.get("PENGU_MIC_DEVICE")
        if env_device:
            try:
                configured = int(env_device)
            except ValueError:
                pass

        manager = AudioDeviceManager(
            configured_device=configured,
            target_sample_rate=TARGET_SAMPLE_RATE,
            num_rounds=3,
        )

        selection = manager.select_best_device()
        if selection is None:
            logger.error("no_usable_microphone")
            return False

        self._selection = selection

        # Open the stream using EXACTLY the parameters validated during probe
        return self._open_stream(selection)

    def _open_stream(self, sel: DeviceSelection) -> bool:
        """
        Open a persistent audio stream using the validated device selection.
        Uses the exact same device, sample rate, and channels from the probe.
        """
        try:
            import sounddevice as sd

            device = sel.device_index
            sr = sel.capture_sample_rate
            ch = sel.capture_channels

            logger.info(
                "stream_opening",
                device=device,
                name=sel.device_name,
                api=sel.host_api,
                sr=sr,
                ch=ch,
                rms=sel.rms,
                score=sel.quality_score,
            )

            # Try opening with validated parameters
            try:
                sd.check_input_settings(device=device, samplerate=sr, channels=ch, dtype="int16")
                self._stream = sd.InputStream(
                    samplerate=sr,
                    channels=ch,
                    dtype="int16",
                    device=device,
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self._stream.start()
                self._is_active = True
                self._device_info = sd.query_devices(device)
                logger.info("stream_started", device=device, sr=sr, ch=ch)
                return True
            except Exception as e:
                logger.error("stream_open_failed", device=device, error=str(e))

            # Fallback: try 16kHz mono
            if sr != TARGET_SAMPLE_RATE or ch != TARGET_CHANNELS:
                try:
                    sd.check_input_settings(device=device, samplerate=TARGET_SAMPLE_RATE, channels=TARGET_CHANNELS, dtype="int16")
                    self._stream = sd.InputStream(
                        samplerate=TARGET_SAMPLE_RATE,
                        channels=TARGET_CHANNELS,
                        dtype="int16",
                        device=device,
                        callback=self._audio_callback,
                        blocksize=1024,
                    )
                    self._stream.start()
                    self._is_active = True
                    self._device_info = sd.query_devices(device)
                    # Update selection to reflect actual stream params
                    self._selection.capture_sample_rate = TARGET_SAMPLE_RATE
                    self._selection.capture_channels = TARGET_CHANNELS
                    logger.info("stream_started_fallback", device=device, sr=TARGET_SAMPLE_RATE, ch=TARGET_CHANNELS)
                    return True
                except Exception as e2:
                    logger.error("stream_fallback_failed", error=str(e2))

            # WDM-KS and some devices don't support InputStream callbacks.
            # Use a thread-based sd.rec() loop as fallback.
            try:
                self._rec_thread_stop = threading.Event()
                self._rec_device = device
                self._rec_sr = sr
                self._rec_ch = ch
                self._rec_thread = threading.Thread(
                    target=self._rec_loop, daemon=True, name="mic-rec-fallback"
                )
                self._rec_thread.start()
                self._is_active = True
                self._device_info = sd.query_devices(device)
                logger.info("stream_started_rec_thread", device=device, sr=sr, ch=ch)
                return True
            except Exception as e3:
                logger.error("stream_all_methods_failed", error=str(e3))

            return False

        except Exception as e:
            logger.error("stream_open_error", error=str(e))
            return False

    def stop(self) -> None:
        self._is_active = False
        if hasattr(self, '_rec_thread_stop'):
            self._rec_thread_stop.set()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """
        Audio stream callback — processes and queues audio.

        Pipeline:
          1. Copy raw audio
          2. Downmix to mono (if multi-channel)
          3. Resample to 16kHz (if device native rate != 16kHz)
          4. Queue for VAD / STT processing
        """
        if not self._is_active or self._is_muted:
            return

        audio = indata.copy()
        audio = _downmix_to_mono(audio)

        sel = self._selection
        if sel and sel.capture_sample_rate != TARGET_SAMPLE_RATE:
            audio = _resample(audio, sel.capture_sample_rate, TARGET_SAMPLE_RATE)

        self._audio_queue.put(audio)

    def _rec_loop(self) -> None:
        """Fallback capture loop using blocking sd.rec() for devices
        that don't support InputStream callbacks (e.g. WDM-KS)."""
        chunk_frames = 1024
        while not self._rec_thread_stop.is_set():
            try:
                audio = sd.rec(
                    chunk_frames,
                    samplerate=self._rec_sr,
                    channels=self._rec_ch,
                    dtype="int16",
                    device=self._rec_device,
                )
                sd.wait()
                if audio is not None and audio.size > 0:
                    if not self._is_muted:
                        a = audio.copy()
                        a = _downmix_to_mono(a)
                        if self._rec_sr != TARGET_SAMPLE_RATE:
                            a = _resample(a, self._rec_sr, TARGET_SAMPLE_RATE)
                        self._audio_queue.put(a)
            except Exception:
                if not self._rec_thread_stop.is_set():
                    time.sleep(0.05)

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def calculate_energy(self, audio: np.ndarray) -> float:
        energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        self._rms_history.append(energy)
        if len(self._rms_history) > 200:
            self._rms_history.pop(0)
        if energy > self._peak_level:
            self._peak_level = energy
        return energy

    def calibrate_noise_floor(self, num_frames: int = 50) -> float:
        if not self._rms_history:
            collected = 0
            while collected < num_frames:
                audio = self.get_audio(timeout=0.1)
                if audio is not None:
                    self.calculate_energy(audio)
                    collected += 1
                else:
                    break
        if self._rms_history:
            sorted_rms = sorted(self._rms_history)
            idx = max(0, len(sorted_rms) // 4)
            self._noise_floor = sorted_rms[idx] if sorted_rms else 0.5
        adaptive_threshold = max(self._noise_floor * 3.0, 5.0)
        self._config.vad_energy_threshold = adaptive_threshold
        self._calibrated = True
        logger.info("noise_calibrated", noise_floor=round(self._noise_floor, 2), threshold=round(adaptive_threshold, 2))
        return adaptive_threshold

    def flush(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def mute(self) -> None:
        with self._lock:
            self._is_muted = True
            self.flush()

    def unmute(self) -> None:
        with self._lock:
            self._is_muted = False

    def check_audio_quality(self) -> tuple[bool, str]:
        """
        Quality gate - checks if the selected device produces acceptable audio.
        Uses the same AudioDeviceManager probe data.
        Produces per-check breakdown showing exactly what passed/failed.
        """
        if not self._is_active or self._selection is None:
            return False, "Microphone is not active."

        sel = self._selection

        report_lines = [
            "MICROPHONE CHECK",
            "",
            f"  Device:       {sel.device_name}",
            f"  API:          {sel.host_api}",
            f"  Sample Rate:  {sel.capture_sample_rate} Hz",
            f"  Channels:     {sel.capture_channels}",
            "",
            f"  Noise Floor:  {sel.noise_floor:.1f}",
            f"  Speech RMS:   {sel.speech_rms:.1f}",
            f"  Peak:         {sel.peak:.1f}",
            f"  SNR:          {sel.snr_db:.1f} dB",
            f"  Clipping:     {sel.clipping_percent:.4f}%",
            f"  Quality:      {sel.quality.value}",
            f"  Score:        {sel.quality_score:.1f}",
            "",
        ]

        # --- Per-check gates ---
        checks = []

        checks.append(("Device opens", "OK", True))
        checks.append(("Signal detected", "OK" if sel.rms > 0 else "FAIL", sel.rms > 0))
        checks.append(("Speech detected", "OK" if sel.speech_detected else "FAIL", sel.speech_detected))
        snr_ok = sel.snr_db >= MIN_SNR_DB
        checks.append((f"SNR >= {MIN_SNR_DB} dB", "OK" if snr_ok else "FAIL", snr_ok))
        rms_ok = sel.speech_rms >= MIN_SPEECH_RMS
        checks.append((f"Speech RMS >= {MIN_SPEECH_RMS}", "OK" if rms_ok else "FAIL", rms_ok))
        clip_ok = sel.clipping_percent < 1.0
        checks.append(("Clipping < 1%", "OK" if clip_ok else "WARN", clip_ok))
        quality_ok = sel.quality.is_voice_ready
        quality_hw = sel.quality.is_hardware_ok
        if quality_ok:
            checks.append(("Quality >= ACCEPTABLE", "OK", True))
        elif quality_hw:
            checks.append(("Hardware OK (no speech observed)", "OK", True))
            checks.append(("Speech evidence", "WARN", False))
        else:
            checks.append(("Quality >= ACCEPTABLE", "FAIL", False))

        all_passed = all(c[2] for c in checks)

        for name, status, _ in checks:
            icon = "[OK]" if status == "OK" else f"[{status}]"
            report_lines.append(f"  {icon} {name}")

        report_lines.append("")

        if all_passed:
            report_lines.append("  Status:       READY")
            full_report = "\n".join(report_lines)
            logger.info("mic_quality_gate_passed", report=full_report)
            return True, full_report

        # --- NOT READY: provide recommendations ---
        report_lines.append("  Status:       NOT READY")
        report_lines.append("")
        failed = [name for name, status, ok in checks if not ok]
        if failed:
            report_lines.append("  WHAT FAILED:")
            for f_name in failed:
                report_lines.append(f"    - {f_name}")

        report_lines.append("")
        best_alt = None
        for p in sel.all_probes:
            if p.get("device") != sel.device_index and p.get("score", 0) > sel.quality_score:
                if best_alt is None or p.get("score", 0) > best_alt.get("score", 0):
                    best_alt = p
        if best_alt:
            report_lines.append("  RECOMMENDED:")
            report_lines.append("    " + best_alt["name"] + " (Device " + str(best_alt["device"]) + ", score=" + str(best_alt["score"]) + ", API=" + best_alt["api"] + ")")

        full_report = "\n".join(report_lines)
        logger.warning("mic_quality_gate_failed", report=full_report)
        return False, full_report

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def selection(self) -> Optional[DeviceSelection]:
        return self._selection

    def get_level(self) -> dict[str, Any]:
        avg_rms = float(np.mean(self._rms_history[-50:])) if self._rms_history else 0.0
        sel = self._selection
        return {
            "active": self._is_active,
            "muted": self._is_muted,
            "avg_rms": round(avg_rms, 2),
            "peak": round(self._peak_level, 2),
            "noise_floor": round(self._noise_floor, 2),
            "calibrated": self._calibrated,
            "device": sel.device_name if sel else "unknown",
            "device_index": sel.device_index if sel else None,
            "host_api": sel.host_api if sel else "unknown",
            "probe_rms": round(sel.rms, 1) if sel else 0,
            "probe_score": round(sel.quality_score, 1) if sel else 0,
            "sample_rate": sel.capture_sample_rate if sel else TARGET_SAMPLE_RATE,
            "target_sample_rate": TARGET_SAMPLE_RATE,
        }


# ---------------------------------------------------------------------------
# Adaptive VAD
# ---------------------------------------------------------------------------

class AdaptiveVAD:
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._speech_detected = False
        self._speech_start_time: float = 0
        self._consecutive_silence_frames: int = 0

    def reset(self) -> None:
        self._speech_detected = False
        self._consecutive_silence_frames = 0

    def process_chunk(self, audio: np.ndarray, energy: float) -> tuple[bool, bool]:
        threshold = self._config.vad_energy_threshold
        if not self._speech_detected:
            if energy > threshold:
                self._speech_detected = True
                self._speech_start_time = time.time()
                self._consecutive_silence_frames = 0
                return True, False
            return False, False
        else:
            speech_duration = time.time() - self._speech_start_time
            if energy < threshold * 0.5:
                self._consecutive_silence_frames += 1
                if self._consecutive_silence_frames >= int(self._config.vad_silence_timeout * 10):
                    self._speech_detected = False
                    self._consecutive_silence_frames = 0
                    return False, True
            else:
                self._consecutive_silence_frames = 0
            if speech_duration > self._config.vad_max_speech_duration:
                self._speech_detected = False
                return False, True
            return True, False


# ---------------------------------------------------------------------------
# Wake Word Detector (STT-based)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    def __init__(self, config: VoiceConfig, stt_engine: "STTEngine", event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._config = config
        self._stt = stt_engine
        self._event_loop = event_loop
        self._last_wake_time: float = 0
        self._speech_active = False
        self._buffer: list[np.ndarray] = []
        self._speech_start: float = 0.0
        self._silence_frames: int = 0

    def process_chunk_simple(self, audio: np.ndarray, energy: float) -> Optional[str]:
        now = time.time()
        if now - self._last_wake_time < self._config.wake_debounce_seconds:
            return None

        if not self._speech_active:
            if energy > self._config.vad_energy_threshold:
                self._speech_active = True
                self._buffer = [audio]
                self._speech_start = now
                self._silence_frames = 0
                logger.debug("wake_speech_start", energy=round(energy, 2), threshold=round(self._config.vad_energy_threshold, 2))
            return None
        else:
            self._buffer.append(audio)
            speech_duration = now - self._speech_start
            if energy < self._config.vad_energy_threshold * 0.5:
                self._silence_frames += 1
                if self._silence_frames >= 3:
                    self._speech_active = False
                    if speech_duration >= 0.3 and self._buffer:
                        logger.debug("wake_speech_end", duration=round(speech_duration, 2), chunks=len(self._buffer))
                        result = self._check_wake_phrase(self._buffer)
                        self._buffer = []
                        if result:
                            self._last_wake_time = time.time()
                            return result
                    self._buffer = []
            else:
                self._silence_frames = 0
            if speech_duration > self._config.vad_max_speech_duration:
                self._speech_active = False
                if speech_duration >= 0.3 and self._buffer:
                    result = self._check_wake_phrase(self._buffer)
                    self._buffer = []
                    if result:
                        self._last_wake_time = time.time()
                        return result
                self._buffer = []
        return None

    def _check_wake_phrase(self, buffer: list[np.ndarray]) -> Optional[str]:
        if not buffer:
            return None
        audio = np.concatenate(buffer)
        duration = len(audio) / self._config.sample_rate
        # Pre-filter: skip utterances unlikely to be wake phrase
        if duration < 0.3 or duration > 10.0:
            return None
        # Quick energy sanity check: skip very quiet utterances
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < 10.0:
            return None
        logger.info("wake_check_transcribing", duration=f"{duration:.2f}s", rms=f"{rms:.1f}")
        try:
            if self._event_loop and self._event_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._stt.transcribe(audio), self._event_loop
                )
                text = future.result(timeout=30)
            else:
                loop = asyncio.new_event_loop()
                try:
                    text = loop.run_until_complete(self._stt.transcribe(audio))
                finally:
                    loop.close()
        except Exception as e:
            logger.error("wake_stt_error", error=str(e))
            return None
        if not text:
            return None
        text_lower = text.lower().strip()
        logger.info("wake_check_text", text=text_lower[:100])
        for phrase in [self._config.wake_phrase.lower(), "hello pengu", "hey pengu", "pengu"]:
            if phrase in text_lower:
                logger.info("wake_phrase_detected", text=text_lower, phrase=phrase)
                return text_lower
        return None

    def reset(self) -> None:
        self._speech_active = False
        self._buffer = []
        self._silence_frames = 0
        self._last_wake_time = 0


# ---------------------------------------------------------------------------
# Command Recorder
# ---------------------------------------------------------------------------

class CommandRecorder:
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._last_speech_time: float = 0
        self._start_time: float = 0
        self._has_speech: bool = False
        self._noise_floor: float = 0.5

    def start(self) -> None:
        self._audio_buffer.clear()
        self._is_recording = True
        self._start_time = time.time()
        self._last_speech_time = 0
        self._has_speech = False

    def add_chunk(self, audio: np.ndarray, energy: float) -> None:
        if not self._is_recording:
            return
        self._audio_buffer.append(audio)
        threshold = max(self._config.vad_energy_threshold, self._noise_floor * 3.0)
        if energy > threshold:
            self._last_speech_time = time.time()
            self._has_speech = True

    def set_noise_floor(self, noise_floor: float) -> None:
        self._noise_floor = noise_floor

    def should_stop(self) -> bool:
        if not self._is_recording:
            return False
        now = time.time()
        if now - self._start_time > self._config.command_max_duration:
            return True
        if self._has_speech and self._last_speech_time > 0:
            if now - self._last_speech_time > self._config.command_silence_timeout:
                return True
        return False

    def stop(self) -> Optional[np.ndarray]:
        self._is_recording = False
        if not self._audio_buffer:
            return None
        audio = np.concatenate(self._audio_buffer)
        duration = len(audio) / self._config.sample_rate
        if duration < self._config.command_min_duration:
            return None
        return audio


# ---------------------------------------------------------------------------
# STT Engine
# ---------------------------------------------------------------------------

class STTEngine:
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._available = False

    async def initialize(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            model_size = os.environ.get("PENGU_STT_MODEL", self._config.stt_model_size)
            compute_type = os.environ.get("PENGU_STT_COMPUTE_TYPE", "int8")
            device = os.environ.get("PENGU_STT_DEVICE", "cpu")
            logger.info("loading_stt_model", model=model_size, device=device, compute_type=compute_type)
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._available = True
            logger.info("stt_model_loaded", model=model_size)
            return True
        except Exception as e:
            logger.error("stt_model_load_failed", error=str(e))
            return False

    async def transcribe(self, audio: np.ndarray) -> Optional[str]:
        if not self._available or self._model is None:
            return None
        try:
            audio_flat = audio.flatten() if audio.ndim > 1 else audio
            audio_float = audio_flat.astype(np.float32) / 32768.0
            start = time.time()
            language = os.environ.get("PENGU_STT_LANGUAGE", self._config.stt_language)
            segments, info = self._model.transcribe(audio_float, language=language, beam_size=5, vad_filter=True)
            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts).strip()
            latency = time.time() - start
            logger.info("stt_complete", text=full_text[:100] if full_text else "(empty)", latency=f"{latency:.2f}s")
            return full_text if full_text else None
        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            return None

    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# TTS Engine
# ---------------------------------------------------------------------------

class TTSEngine:
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._available = False
        self._is_speaking = False
        self._cancel_event = threading.Event()
        self._speak_lock = threading.Lock()
        self._process = None

    async def initialize(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            self._available = True
            return True
        except ImportError:
            return False

    async def speak(self, text: str) -> bool:
        if not self._available or not text.strip():
            return False
        with self._speak_lock:
            self._is_speaking = True
            self._cancel_event.clear()
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self._config.tts_voice, rate=self._config.tts_rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if self._cancel_event.is_set():
                    return False
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            if not audio_data:
                return False
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            try:
                import subprocess
                ps_cmd = f'''
                Add-Type -AssemblyName PresentationCore
                $player = New-Object System.Windows.Media.MediaPlayer
                $player.Open([uri]::new("{temp_path}"))
                $player.Play()
                Start-Sleep -Milliseconds 500
                while ($player.Position -lt $player.NaturalDuration.TimeSpan) {{
                    if ([System.Threading.EventWaitHandle]::WaitOne(100)) {{ break }}
                    Start-Sleep -Milliseconds 100
                }}
                $player.Close()
                '''
                self._process = subprocess.Popen(["powershell", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                start = time.time()
                while self._process.poll() is None:
                    if self._cancel_event.is_set():
                        try: self._process.kill()
                        except: pass
                        return False
                    if time.time() - start > 30:
                        try: self._process.kill()
                        except: pass
                        break
                    time.sleep(0.1)
            finally:
                self._process = None
                try: os.unlink(temp_path)
                except OSError: pass
            logger.info("tts_spoke", text_length=len(text))
            return True
        except Exception as e:
            logger.error("tts_failed", error=str(e))
            return False
        finally:
            with self._speak_lock:
                self._is_speaking = False
                self._process = None

    def cancel(self) -> None:
        self._cancel_event.set()
        if self._process:
            try: self._process.kill()
            except: pass

    def is_available(self) -> bool:
        return self._available

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


# ---------------------------------------------------------------------------
# Voice Engine
# ---------------------------------------------------------------------------

class VoiceEngine:
    """
    Production voice engine.

    Device selection: delegated to AudioDeviceManager (single source of truth).
    """

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        command_callback: Optional[Callable[[str], Optional[str]]] = None,
        state_callback: Optional[Callable[[VoiceState], None]] = None,
    ) -> None:
        self._config = config or VoiceConfig()
        self._command_callback = command_callback
        self._state_callback = state_callback
        self._microphone = MicrophoneManager(self._config)
        self._stt = STTEngine(self._config)
        self._tts = TTSEngine(self._config)
        # Event loop is set in start() � pass None initially, set later
        self._wake_detector = WakeWordDetector(self._config, self._stt, event_loop=None)
        self._command_recorder = CommandRecorder(self._config)
        self._state = VoiceState.OFFLINE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pending_command_audio: Optional[np.ndarray] = None
        self._mic_quality_ok = False
        self._diagnostics: dict[str, Any] = {
            "wake_detections": 0, "commands_processed": 0,
            "empty_transcriptions": 0, "tts_spoken": 0, "errors": 0,
        }

    async def initialize(self) -> dict[str, bool]:
        self._set_state(VoiceState.STARTING)
        stt_ok = await self._stt.initialize()
        tts_ok = await self._tts.initialize()
        mic_ok = self._microphone.select_and_start()
        result = {"stt": stt_ok, "tts": tts_ok, "microphone": mic_ok}
        logger.info("voice_init", **result)

        if mic_ok:
            self._mic_quality_ok, quality_report = self._microphone.check_audio_quality()
            result["mic_quality"] = self._mic_quality_ok
            if not self._mic_quality_ok:
                logger.warning("mic_quality_gate_failed", report=quality_report)
            self._microphone.calibrate_noise_floor()
            self._command_recorder.set_noise_floor(self._microphone.noise_floor)

            # Validate the selected mic actually produces usable audio for STT
            if stt_ok:
                stt_valid = await self._validate_mic_with_stt()
                result["mic_stt_valid"] = stt_valid
                if not stt_valid:
                    logger.warning("mic_stt_validation_failed", hint="Device audio may produce poor STT results")

        return result

    async def _validate_mic_with_stt(self) -> bool:
        """Capture a short audio sample and verify STT can process it.

        Returns True if the device produces audio that STT can handle
        (even if no speech is present — we just check the pipeline works).
        Returns False if the audio is completely broken.
        """
        try:
            # Collect 2 seconds of audio from the running stream
            chunks = []
            start = time.time()
            while time.time() - start < 2.0:
                audio = self._microphone.get_audio(timeout=0.2)
                if audio is not None:
                    chunks.append(audio)
            if not chunks:
                logger.warning("mic_stt_validation_no_audio")
                return False
            audio = np.concatenate(chunks)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            logger.info("mic_stt_validation", duration=f"{len(audio)/self._config.sample_rate:.1f}s", rms=round(rms, 1))
            # Run STT in a thread to avoid event-loop conflicts.
            # The caller (PenguApp.start) already has an asyncio loop running.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                def _run_stt():
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(self._stt.transcribe(audio))
                    finally:
                        loop.close()
                future = pool.submit(_run_stt)
                text = future.result(timeout=15)
            logger.info("mic_stt_validation_result", text=text[:100] if text else "(empty)")
            # Empty transcription is OK — means the pipeline works, just no speech
            # We only fail if STT throws an exception or produces garbage
            return True
        except Exception as e:
            logger.error("mic_stt_validation_error", error=str(e))
            return False

    async def start(self) -> None:
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._wake_detector._event_loop = self._loop
        # Run the event loop in a dedicated thread so run_coroutine_threadsafe works.
        self._loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="voice-event-loop",
        )
        self._loop_thread.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="voice-engine")
        self._thread.start()
        self._set_state(VoiceState.STANDBY)
        logger.info("voice_engine_started")

    async def stop(self) -> None:
        self._running = False
        self._set_state(VoiceState.STOPPING)
        self._tts.cancel()
        self._microphone.stop()
        # Stop the event loop thread
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._set_state(VoiceState.OFFLINE)
        logger.info("voice_engine_stopped")

    def _run_event_loop(self) -> None:
        """Target for the event-loop thread: runs the loop until stopped."""
        try:
            self._loop.run_forever()
        finally:
            if not self._loop.is_closed():
                self._loop.close()

    def _run_loop(self) -> None:
        consecutive_errors = 0
        while self._running:
            try:
                consecutive_errors = 0
                self._phase_standby_and_detect()
                if not self._running: break
                self._phase_acknowledge()
                if not self._running: break
                self._phase_listen_command()
                if not self._running: break
                self._phase_transcribe_and_execute()
            except Exception as e:
                logger.error("voice_loop_error", error=str(e), exc_info=True)
                self._diagnostics["errors"] += 1
                consecutive_errors += 1
                self._set_state(VoiceState.ERROR)
                time.sleep(5 if consecutive_errors >= 5 else 1)
                if self._running:
                    self._set_state(VoiceState.STANDBY)

    def _phase_standby_and_detect(self) -> None:
        self._set_state(VoiceState.STANDBY)
        self._wake_detector.reset()
        self._microphone.flush()
        self._microphone.unmute()
        threshold = self._config.vad_energy_threshold
        logger.info("standby_listening", threshold=round(threshold, 2))
        _debug_counter = 0
        while self._running:
            audio = self._microphone.get_audio(timeout=0.1)
            if audio is None:
                continue
            energy = self._microphone.calculate_energy(audio)
            # Periodic diagnostic: show energy vs threshold
            _debug_counter += 1
            if _debug_counter % 50 == 0:
                logger.debug(
                    "voice_loop_energy",
                    energy=round(energy, 2),
                    threshold=round(threshold, 2),
                    speech_active=self._wake_detector._speech_active,
                    queue_size=self._microphone._audio_queue.qsize(),
                )
            wake = self._wake_detector.process_chunk_simple(audio, energy)
            if wake is not None:
                self._diagnostics["wake_detections"] += 1
                logger.info("wake_detected", text=wake)
                return

    def _phase_acknowledge(self) -> None:
        self._set_state(VoiceState.WAKE_DETECTED)
        time.sleep(0.2)
        self._set_state(VoiceState.ACKNOWLEDGING)
        self._microphone.mute()
        try:
            future = asyncio.run_coroutine_threadsafe(self._tts.speak("Yes?"), self._loop)
            future.result(timeout=10)
        except Exception as e:
            logger.warning("tts_acknowledge_failed", error=str(e))
        time.sleep(0.4)
        self._microphone.flush()
        self._microphone.unmute()

    def _phase_listen_command(self) -> None:
        self._set_state(VoiceState.LISTENING)
        self._command_recorder.start()
        logger.info("command_listening")
        while self._running and not self._command_recorder.should_stop():
            audio = self._microphone.get_audio(timeout=0.1)
            if audio is not None:
                energy = self._microphone.calculate_energy(audio)
                self._command_recorder.add_chunk(audio, energy)
        speech_audio = self._command_recorder.stop()
        if speech_audio is None:
            logger.info("no_command_detected")
            self._microphone.mute()
            self._safe_tts_speak("I didn't hear a command.")
            time.sleep(0.3)
            self._microphone.unmute()
            return
        self._pending_command_audio = speech_audio

    def _phase_transcribe_and_execute(self) -> None:
        if self._pending_command_audio is None: return
        speech_audio = self._pending_command_audio
        self._pending_command_audio = None
        self._set_state(VoiceState.TRANSCRIBING)
        try:
            future = asyncio.run_coroutine_threadsafe(self._stt.transcribe(speech_audio), self._loop)
            text = future.result(timeout=30)
        except Exception as e:
            logger.error("transcription_error", error=str(e))
            text = None
        if not text:
            self._diagnostics["empty_transcriptions"] += 1
            self._microphone.mute()
            self._safe_tts_speak("Sorry, I didn't catch that. Please try again.")
            time.sleep(0.3)
            self._microphone.unmute()
            return
        logger.info("command_transcribed", text=text)
        self._set_state(VoiceState.THINKING)
        result_text = None
        if self._command_callback:
            try:
                self._set_state(VoiceState.EXECUTING)
                result_text = self._command_callback(text)
                self._diagnostics["commands_processed"] += 1
            except Exception as e:
                logger.error("command_error", error=str(e))
                result_text = f"Error executing command: {e}"
        if result_text:
            self._set_state(VoiceState.SPEAKING)
            self._microphone.mute()
            self._safe_tts_speak(result_text)
            self._diagnostics["tts_spoken"] += 1
            time.sleep(0.3)
            self._microphone.unmute()

    def _safe_tts_speak(self, text: str) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(self._tts.speak(text), self._loop)
            future.result(timeout=30)
        except Exception as e:
            logger.warning("tts_speak_failed", text_length=len(text), error=str(e))

    def interrupt(self) -> None:
        self._tts.cancel()
        self._microphone.flush()

    def _set_state(self, state: VoiceState) -> None:
        old = self._state
        self._state = state
        if state != old:
            logger.info("state_transition", from_state=old.value, to_state=state.value)
        if self._state_callback:
            try: self._state_callback(state)
            except: pass

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running, "state": self._state.value,
            "stt_available": self._stt.is_available(),
            "tts_available": self._tts.is_available(),
            "tts_speaking": self._tts.is_speaking,
            "microphone": self._microphone.get_level(),
            "mic_quality_ok": self._mic_quality_ok,
            "diagnostics": self._diagnostics.copy(),
            "config": {
                "wake_phrase": self._config.wake_phrase,
                "stt_model": self._config.stt_model_size,
                "tts_voice": self._config.tts_voice,
                "vad_threshold": self._config.vad_energy_threshold,
            },
        }

    def run_diagnostics(self) -> dict[str, Any]:
        mic_level = self._microphone.get_level()
        sel = self._microphone.selection
        return {
            "microphone": {
                "status": "OK" if mic_level["active"] else "NOT ACTIVE",
                "device": mic_level["device"],
                "device_index": mic_level["device_index"],
                "host_api": mic_level.get("host_api", "unknown"),
                "probe_rms": mic_level.get("probe_rms", 0),
                "probe_score": mic_level.get("probe_score", 0),
                "sample_rate": mic_level["sample_rate"],
                "avg_rms": mic_level["avg_rms"],
                "peak": mic_level["peak"],
                "noise_floor": mic_level["noise_floor"],
                "calibrated": mic_level["calibrated"],
                "quality_ok": self._mic_quality_ok,
                "quality": sel.quality.value if sel else "unknown",
                "snr_db": round(sel.snr_db, 1) if sel else 0,
                "speech_rms": round(sel.speech_rms, 1) if sel else 0,
            },
            "stt": {"status": "OK" if self._stt.is_available() else "NOT LOADED", "model": self._config.stt_model_size},
            "tts": {"status": "OK" if self._tts.is_available() else "NOT AVAILABLE", "voice": self._config.tts_voice},
            "wake_word": {"status": "OK", "phrase": self._config.wake_phrase, "method": "VAD + STT transcription"},
            "stats": self._diagnostics.copy(),
        }
