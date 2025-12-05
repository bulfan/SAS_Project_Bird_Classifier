from typing import Optional, Union
import numpy as np
from pydub import AudioSegment

"""
root_means_sqaured.py

Utilities to compute RMS for MP3 files.

Requires: pydub, numpy
pydub requires ffmpeg/avlib available on the system.

Example:
    # overall RMS (single float)
    r = rms_mp3("audio.mp3")
    # frame-wise RMS every 100 ms, returned as numpy array of shape (frames, channels) or (frames,) for mono
    frames = rms_mp3("audio.mp3", frame_ms=100)
"""



def _samples_from_segment(seg: AudioSegment) -> (np.ndarray, int):
    samples = np.array(seg.get_array_of_samples())
    channels = seg.channels
    if channels > 1:
        samples = samples.reshape((-1, channels))
    # convert to float32 in range [-1.0, 1.0)
    sample_width = seg.sample_width * 8  # bits per sample
    max_val = float(2 ** (sample_width - 1))
    samples = samples.astype(np.float32) / max_val
    return samples, channels


def _rms_array(samples: np.ndarray, axis=0) -> np.ndarray:
    # samples: shape (n,) or (n, channels)
    return np.sqrt(np.mean(np.square(samples), axis=axis))


def rms_mp3(path: str,
            frame_ms: Optional[int] = None,
            to_db: bool = False,
            eps: float = 1e-12) -> Union[float, np.ndarray]:
    """
    Compute RMS of an MP3 file.

    Args:
        path: path to the mp3 file.
        frame_ms: if None, return a single RMS value for the whole file.
                  If an integer (milliseconds), return RMS for each non-overlapping frame of that duration.
        to_db: if True, convert RMS to dBFS (0 dB = full scale). Returns negative values (or -inf for silence).
        eps: small value to avoid log(0).

    Returns:
        If frame_ms is None:
            - single float (mono) or 1D numpy array (per-channel) with RMS in linear units (0..1) or dB.
        If frame_ms is set:
            - numpy array of shape (frames, channels) for multi-channel or (frames,) for mono.
    """
    seg = AudioSegment.from_file(path, format="mp3")
    if frame_ms is None:
        samples, channels = _samples_from_segment(seg)
        if channels == 1:
            rms = float(_rms_array(samples, axis=0))
        else:
            rms = _rms_array(samples, axis=0)  # per-channel
    else:
        # split into frames
        frames = []
        for start in range(0, len(seg), frame_ms):
            frame_seg = seg[start:start + frame_ms]
            frame_samples, ch = _samples_from_segment(frame_seg)
            # ensure consistent channels
            if frame_samples.size == 0:
                # empty frame -> zeros
                shape = (0, ch) if ch > 1 else (0,)
                frame_rms = np.zeros((1, ch)) if ch > 1 else np.zeros((1,))
            else:
                frame_rms = _rms_array(frame_samples, axis=0)
                # make 1D -> shape (1, channels) for stacking
                if ch > 1:
                    frame_rms = frame_rms.reshape((1, ch))
                else:
                    frame_rms = np.array([frame_rms])
            frames.append(frame_rms)
        if not frames:
            return np.array([])  # empty file
        stacked = np.vstack(frames)
        # If mono, return 1D array
        if stacked.shape[1] == 1:
            rms = stacked[:, 0]
        else:
            rms = stacked

    if to_db:
        # convert linear RMS to dBFS
        if isinstance(rms, np.ndarray):
            return 20.0 * np.log10(np.maximum(rms, eps))
        else:
            return float(20.0 * np.log10(max(rms, eps)))
    return rms


    if __name__ == "__main__":
        # Example usage
        mp3_file = r"C:\Users\jpgft\Downloads\SaS\train-bird-audio\train-bird-audio\pygnut\XC350753.mp3"
        
        # Overall RMS (single float)
        r = rms_mp3(mp3_file)
        print(f"Overall RMS: {r}")
        
        # Frame-wise RMS every 100 ms
        frames = rms_mp3(mp3_file, frame_ms=100)
        print(f"Frame-wise RMS shape: {frames.shape}")
        print(f"First few frames: {frames[:5]}")
        
        # Frame-wise RMS in dB
        frames_db = rms_mp3(mp3_file, frame_ms=100, to_db=True)
        print(f"Frame-wise RMS (dB) first few frames: {frames_db[:5]}")
        