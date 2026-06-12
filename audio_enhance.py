"""
Audio Enhancement Pipeline for Voice AI
========================================
Post-processing module that brings raw TTS output to broadcast/studio quality.
Applied automatically to all local TTS synthesis before returning to client.
"""

import numpy as np
import librosa
from scipy import signal

__all__ = ["enhance_audio", "crossfade_segments", "add_natural_pause"]


def enhance_audio(wav_data, sample_rate, target_sr=48000):
    """
    Main enhancement pipeline. Takes raw TTS numpy audio → returns studio-quality audio.
    
    Pipeline:
      1. Resample to target_sr (high-quality soxr)
      2. Remove DC offset
      3. High-pass filter (60Hz) to remove rumble
      4. Noise gate (clean silence regions)
      5. De-harsh AI artifacts (gentle 5-8kHz attenuation)
      6. Peak normalization + soft limiter
      7. Trim leading/trailing silence
    """
    if wav_data is None or len(wav_data) == 0:
        return np.zeros(int(target_sr * 0.1), dtype=np.float32)

    # Ensure float32 mono
    wav = np.asarray(wav_data, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    # Remove NaN/Inf
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    # Skip processing for very short audio
    if len(wav) < int(sample_rate * 0.05):
        if sample_rate != target_sr:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr, res_type='soxr_hq')
        return wav

    try:
        # 1. Resample to target sample rate
        if sample_rate != target_sr:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr, res_type='soxr_hq')
            print(f"[Voice AI Enhance] Resampled {sample_rate}Hz → {target_sr}Hz")
        sr = target_sr

        # 2. Remove DC offset
        wav = wav - np.mean(wav)

        # 3. High-pass filter at 60Hz (remove rumble/hum)
        try:
            sos_hp = signal.butter(2, 60, btype='highpass', fs=sr, output='sos')
            wav = signal.sosfilt(sos_hp, wav).astype(np.float32)
        except Exception as e:
            print(f"[Voice AI Enhance] HP filter skipped: {e}")

        # 4. Noise gate — zero out very quiet samples
        gate_threshold = 0.003
        envelope = np.abs(wav)
        # Smooth envelope with a short window to avoid chopping mid-syllable
        win_size = int(sr * 0.02)  # 20ms window
        if win_size > 1 and len(envelope) > win_size:
            kernel = np.ones(win_size) / win_size
            smooth_env = np.convolve(envelope, kernel, mode='same')
            gate_mask = smooth_env > gate_threshold
            wav = wav * gate_mask.astype(np.float32)
        
        # 5. Gentle de-harsh: attenuate 5-8kHz AI artifacts by ~3dB
        try:
            # Parametric notch centered at 6.5kHz, Q=0.5 (wide), -3dB
            if sr >= 16000:  # Only if sample rate supports this range
                nyq = sr / 2.0
                center_freq = min(6500, nyq * 0.8)  # Don't exceed Nyquist
                # Design a gentle peaking EQ (inverted = cut)
                b_notch, a_notch = signal.iirpeak(center_freq / nyq, 0.5)
                # Apply inverted (cut instead of boost) — mix with dry signal
                wet = signal.lfilter(b_notch, a_notch, wav).astype(np.float32)
                # Blend: 70% dry + 30% inverted-wet = gentle -3dB cut
                wav = 0.7 * wav + 0.3 * (wav - (wet - wav))
                wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            print(f"[Voice AI Enhance] De-harsh skipped: {e}")

        # 6. Peak normalization to -1dB + soft limiter
        peak = np.abs(wav).max()
        if peak > 0.001:
            target_peak = 0.89  # -1dB in linear
            wav = wav * (target_peak / peak)
        
        # Soft knee limiter (tanh-based)
        wav = np.tanh(wav * 1.2) / np.tanh(1.2)
        wav = wav.astype(np.float32)

        # 7. Trim leading/trailing silence (keep 50ms buffer)
        buffer_samples = int(sr * 0.05)
        trim_threshold = 0.01
        abs_wav = np.abs(wav)
        
        # Find first/last sample above threshold
        above = np.where(abs_wav > trim_threshold)[0]
        if len(above) > 0:
            start = max(0, above[0] - buffer_samples)
            end = min(len(wav), above[-1] + buffer_samples)
            wav = wav[start:end]

        print(f"[Voice AI Enhance] Done: {len(wav)} samples, peak={np.abs(wav).max():.3f}")
        return wav

    except Exception as e:
        print(f"[Voice AI Enhance] Pipeline error, returning original: {e}")
        if sample_rate != target_sr:
            try:
                wav_data = librosa.resample(
                    np.asarray(wav_data, dtype=np.float32),
                    orig_sr=sample_rate, target_sr=target_sr, res_type='soxr_hq'
                )
            except Exception:
                pass
        return np.asarray(wav_data, dtype=np.float32)


def crossfade_segments(segments, sr, crossfade_ms=50):
    """
    Join audio segments with smooth crossfade overlap instead of hard concatenation.
    
    Args:
        segments: List of numpy arrays (float32 audio)
        sr: Sample rate
        crossfade_ms: Crossfade duration in milliseconds
    
    Returns:
        Single numpy array with all segments blended together
    """
    if not segments:
        return np.zeros(int(sr * 0.1), dtype=np.float32)
    
    if len(segments) == 1:
        return np.asarray(segments[0], dtype=np.float32)

    crossfade_samples = int(sr * crossfade_ms / 1000.0)
    
    # Build output by overlapping segments
    result = np.asarray(segments[0], dtype=np.float32).copy()
    
    for i in range(1, len(segments)):
        seg = np.asarray(segments[i], dtype=np.float32)
        
        if len(seg) == 0:
            continue
            
        # Actual crossfade length = min of crossfade_samples, half of either segment
        cf_len = min(crossfade_samples, len(result) // 2, len(seg) // 2)
        
        if cf_len < 2:
            # Too short for crossfade, just concatenate
            result = np.concatenate([result, seg])
            continue
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, cf_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, cf_len, dtype=np.float32)
        
        # Apply fades to overlap region
        overlap_out = result[-cf_len:] * fade_out
        overlap_in = seg[:cf_len] * fade_in
        blended = overlap_out + overlap_in
        
        # Build new result: everything before overlap + blended + rest of new segment
        result = np.concatenate([
            result[:-cf_len],
            blended,
            seg[cf_len:]
        ])
    
    return result


def add_natural_pause(sr, punctuation=""):
    """
    Generate a natural-sounding pause (silence with tiny noise floor).
    
    The noise floor prevents the eerie "dead silence" that sounds unnatural
    in synthesized speech. Real recordings always have ambient room tone.
    
    Args:
        sr: Sample rate
        punctuation: The punctuation mark that caused this pause
    
    Returns:
        numpy array of the pause audio
    """
    # Pause durations tuned for natural Vietnamese speech
    pause_map = {
        ",": 0.28,
        ";": 0.28,
        ":": 0.28,
        ".": 0.55,
        "!": 0.45,
        "?": 0.50,
        "...": 0.70,
    }
    
    pause_secs = pause_map.get(punctuation, 0.35)
    num_samples = int(sr * pause_secs)
    
    # Generate very quiet noise floor (-60dB) for natural feel
    noise_amplitude = 0.001  # ~-60dB
    pause = np.random.randn(num_samples).astype(np.float32) * noise_amplitude
    
    # Apply gentle fade in/out to the noise to avoid clicks
    fade_len = min(int(sr * 0.01), num_samples // 4)  # 10ms fade
    if fade_len > 1:
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        pause[:fade_len] *= fade_in
        pause[-fade_len:] *= fade_out
    
    return pause
