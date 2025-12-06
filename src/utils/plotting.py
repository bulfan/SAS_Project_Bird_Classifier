"""
Plotting utilities for audio waveforms.

This module provides a small helper to plot time-domain waveforms
for audio files (MP3/WAV/etc.) using librosa + matplotlib.
"""

from typing import Optional, Tuple

import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(file_path: str,
                  sr: int = 22050,
                  ax: Optional[plt.Axes] = None,
                  show: bool = True,
                  out_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Load an audio file and plot its time-domain waveform.

    Args:
        file_path: Path to audio file.
        sr: Target sample rate for loading.
        ax: Optional matplotlib Axes to plot into. If None, a new figure is created.
        show: Whether to call `plt.show()` after plotting.
        out_path: If provided, save the figure to this path.

    Returns:
        (fig, ax) tuple where fig is the matplotlib Figure and ax is the Axes.
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)

    time = np.arange(len(audio)) / float(sample_rate)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(time, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform: {file_path}")
    ax.set_xlim(0, time[-1] if len(time) > 0 else 0)
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path)

    if show and created_fig:
        plt.show()

    return fig, ax
