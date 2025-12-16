"""
Call detection utilities: detect_calls returns per-call start/end times and durations.
"""
from typing import List, Tuple, Optional
import numpy as np
import librosa

from .audio_processor import bandpass_filter, normalize_audio


def detect_calls(y: np.ndarray,
                 sr: int,
                 top_db: float = 40.0,
                 frame_length: int = 4096,
                 hop_length: int = 1024,
                 min_call_duration: float = 0.05,
                 min_silence_duration: float = 0.05,
                 low_freq: Optional[float] = 1000.0,
                 high_freq: Optional[float] = 8000.0,
                 normalize: bool = True,
                 normalize_method: str = 'rms',
                 normalize_target_rms: float = 0.1) -> List[Tuple[float, float, float]]:
    """
    Detect non-silent segments (calls) in a signal.

    Returns a list of tuples: (start_time_s, end_time_s, duration_s)
    """
    if y is None or len(y) == 0:
        return []

    y = np.asarray(y, dtype=float)

    # Optionally normalize first so detection thresholds are consistent
    if normalize:
        try:
            y = normalize_audio(y, method=normalize_method, target_rms=normalize_target_rms)
        except Exception:
            pass

    # Optional bandpass filtering to focus on expected call band
    if low_freq is not None and high_freq is not None and low_freq < high_freq:
        try:
            y = bandpass_filter(y, sr, low_freq, high_freq)
        except Exception:
            pass

    # Use librosa to split on silence
    intervals = librosa.effects.split(y, top_db=float(top_db), frame_length=frame_length, hop_length=hop_length)

    # Convert to seconds and filter/merge
    segs: List[Tuple[int, int]] = []
    for s, e in intervals:
        dur = (e - s) / float(sr)
        if dur >= min_call_duration:
            segs.append((s, e))

    if not segs:
        return []

    # Merge segments separated by short silent gaps
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = segs[0]
    min_gap_samples = int(max(1, min_silence_duration * sr))
    for s, e in segs[1:]:
        gap = s - cur_e
        if gap <= min_gap_samples:
            # extend
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Convert merged to time tuples
    result: List[Tuple[float, float, float]] = []
    for s, e in merged:
        start_t = s / float(sr)
        end_t = e / float(sr)
        dur = end_t - start_t
        if dur >= min_call_duration:
            result.append((start_t, end_t, dur))

    return result
