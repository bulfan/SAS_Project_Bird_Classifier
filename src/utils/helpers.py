"""
Helper utility functions.
"""

import os


def ensure_dir(path: str):
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def list_audio_files(directory: str, extensions: tuple = ('.wav', '.mp3', '.flac')):
    """
    List all audio files in a directory.

    Args:
        directory: Path to the directory.
        extensions: Tuple of valid audio file extensions.

    Returns:
        List of audio file paths.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files
