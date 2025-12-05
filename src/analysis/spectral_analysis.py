"""
Spectral Analysis Pipeline for Bird Sound Classification.

Frequency-domain analysis of audio signals using Fourier Transform.
Implements all algorithms from scratch (no numpy.fft or similar).

Metrics computed:
- Power Spectral Density (PSD)
- Spectrogram
- Average Power
- Max/Min Frequency
- Max/Min Magnitude (Power)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.data.dataset import BirdSoundDataset


class SpectralAnalysisPipeline:
    """
    Pipeline for frequency-domain spectral analysis of audio signals.
    
    All spectral methods are implemented from scratch, including
    the Discrete Fourier Transform (DFT), without using np.fft or similar.
    """
    
    def __init__(self, cfg: Optional[Any] = None):
        """
        Initialize the Spectral Analysis Pipeline.
        
        Args:
            cfg: Hydra configuration object or None for defaults.
        """
        if cfg is not None:
            spectral_cfg = cfg.get('analysis', {}).get('spectral', {})
            self.n_fft = spectral_cfg.get('n_fft', 2048)
            self.hop_length = spectral_cfg.get('hop_length', 512)
            self.window_type = spectral_cfg.get('window', 'hann')
            self.output_dir = Path(spectral_cfg.get('output_dir', 'outputs/frequency'))
            self.save_plots = spectral_cfg.get('save_plots', True)
            self.show_plots = spectral_cfg.get('show_plots', False)
            self.print_details = spectral_cfg.get('print_details', True)
            self.enabled = spectral_cfg.get('enabled', True)
            self.run_mode = spectral_cfg.get('run_mode', 'single')
            self.single_file = spectral_cfg.get('single_file', None)
            self.folder_path = spectral_cfg.get('folder_path', None)
            self.sample_rate = cfg.get('data', {}).get('sample_rate', 22050)
        else:
            self.n_fft = 2048
            self.hop_length = 512
            self.window_type = 'hann'
            self.output_dir = Path('outputs/frequency')
            self.save_plots = True
            self.show_plots = False
            self.print_details = True
            self.enabled = True
            self.run_mode = 'single'
            self.single_file = None
            self.folder_path = None
            self.sample_rate = 22050
    
    # =========================================================================
    # Audio Loading Utilities
    # =========================================================================
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to normalized numpy array."""
        seg = AudioSegment.from_file(file_path)
        samples = np.array(seg.get_array_of_samples())
        
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
            samples = samples.mean(axis=1)
        
        sample_width_bits = seg.sample_width * 8
        max_val = float(2 ** (sample_width_bits - 1))
        samples = samples.astype(np.float32) / max_val
        
        return samples, seg.frame_rate
    
    # =========================================================================
    # Window Functions (Implemented from Scratch)
    # =========================================================================
    
    def _create_window(self, size: int) -> np.ndarray:
        """Create a window function from scratch."""
        if self.window_type == 'none' or self.window_type is None:
            return np.ones(size, dtype=np.float32)
        
        window = np.zeros(size, dtype=np.float32)
        
        if self.window_type == 'hann':
            for n in range(size):
                window[n] = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size - 1)))
        elif self.window_type == 'hamming':
            for n in range(size):
                window[n] = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (size - 1))
        else:
            window = np.ones(size, dtype=np.float32)
        
        return window
    
    # =========================================================================
    # Discrete Fourier Transform (Implemented from Scratch)
    # =========================================================================
    
    def dft(self, signal: np.ndarray) -> np.ndarray:
        """Compute the Discrete Fourier Transform from scratch (O(N^2))."""
        n = len(signal)
        result = np.zeros(n, dtype=np.complex128)
        
        for k in range(n):
            for t in range(n):
                angle = -2.0 * np.pi * k * t / n
                real_part = np.cos(angle)
                imag_part = np.sin(angle)
                result[k] += signal[t] * (real_part + 1j * imag_part)
        
        return result
    
    def fft(self, signal: np.ndarray) -> np.ndarray:
        """Compute FFT using Cooley-Tukey algorithm (radix-2 DIT)."""
        n = len(signal)
        
        if n & (n - 1) != 0:
            next_pow2 = 1
            while next_pow2 < n:
                next_pow2 <<= 1
            padded = np.zeros(next_pow2, dtype=np.float64)
            padded[:n] = signal
            signal = padded
            n = next_pow2
        
        if n == 1:
            return np.array([signal[0]], dtype=np.complex128)
        
        even = self.fft(signal[0::2])
        odd = self.fft(signal[1::2])
        
        result = np.zeros(n, dtype=np.complex128)
        half_n = n // 2
        
        for k in range(half_n):
            angle = -2.0 * np.pi * k / n
            twiddle = np.cos(angle) + 1j * np.sin(angle)
            result[k] = even[k] + twiddle * odd[k]
            result[k + half_n] = even[k] - twiddle * odd[k]
        
        return result
    
    # =========================================================================
    # Spectral Analysis Methods
    # =========================================================================
    
    def compute_magnitude_spectrum(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the magnitude spectrum of the signal."""
        n = min(len(samples), self.n_fft)
        windowed = samples[:n] * self._create_window(n)
        
        if len(windowed) < self.n_fft:
            padded = np.zeros(self.n_fft)
            padded[:len(windowed)] = windowed
            windowed = padded
        
        spectrum = self.fft(windowed)
        n_positive = len(spectrum) // 2 + 1
        spectrum = spectrum[:n_positive]
        
        magnitudes = np.zeros(n_positive)
        for i in range(n_positive):
            real = spectrum[i].real
            imag = spectrum[i].imag
            magnitudes[i] = np.sqrt(real * real + imag * imag)
        
        frequencies = np.zeros(n_positive)
        freq_resolution = self.sample_rate / self.n_fft
        for i in range(n_positive):
            frequencies[i] = i * freq_resolution
        
        return frequencies, magnitudes
    
    def compute_power_spectral_density(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density (PSD) of the signal."""
        frequencies, magnitudes = self.compute_magnitude_spectrum(samples)
        
        n = len(samples)
        psd = np.zeros(len(magnitudes))
        for i in range(len(magnitudes)):
            psd[i] = (magnitudes[i] * magnitudes[i]) / n
        
        return frequencies, psd
    
    def compute_spectrogram(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram using Short-Time Fourier Transform (STFT)."""
        n_samples = len(samples)
        window = self._create_window(self.n_fft)
        
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length
        if n_frames < 1:
            n_frames = 1
        
        n_freqs = self.n_fft // 2 + 1
        spectrogram = np.zeros((n_freqs, n_frames))
        
        for frame_idx in range(n_frames):
            start = frame_idx * self.hop_length
            end = start + self.n_fft
            
            if end <= n_samples:
                frame = samples[start:end] * window
            else:
                frame = np.zeros(self.n_fft)
                available = n_samples - start
                if available > 0:
                    frame[:available] = samples[start:start + available]
                frame = frame * window
            
            spectrum = self.fft(frame)
            
            for i in range(n_freqs):
                real = spectrum[i].real
                imag = spectrum[i].imag
                spectrogram[i, frame_idx] = np.sqrt(real * real + imag * imag)
        
        times = np.zeros(n_frames)
        for i in range(n_frames):
            times[i] = (i * self.hop_length + self.n_fft / 2) / self.sample_rate
        
        frequencies = np.zeros(n_freqs)
        freq_resolution = self.sample_rate / self.n_fft
        for i in range(n_freqs):
            frequencies[i] = i * freq_resolution
        
        return times, frequencies, spectrogram
    
    def compute_average_power(self, samples: np.ndarray) -> float:
        """Compute average power in the frequency domain."""
        _, psd = self.compute_power_spectral_density(samples)
        
        total = 0.0
        for val in psd:
            total += val
        
        return float(total / len(psd)) if len(psd) > 0 else 0.0
    
    def compute_frequency_range(self, samples: np.ndarray, threshold_db: float = -60.0) -> Tuple[float, float]:
        """Compute min and max frequency with significant energy."""
        frequencies, magnitudes = self.compute_magnitude_spectrum(samples)
        
        if len(magnitudes) == 0:
            return 0.0, 0.0
        
        max_mag = magnitudes[0]
        for mag in magnitudes:
            if mag > max_mag:
                max_mag = mag
        
        if max_mag == 0:
            return 0.0, 0.0
        
        threshold_linear = max_mag * (10.0 ** (threshold_db / 20.0))
        
        min_freq = frequencies[-1]
        max_freq = frequencies[0]
        
        for i, mag in enumerate(magnitudes):
            if mag >= threshold_linear:
                if frequencies[i] < min_freq:
                    min_freq = frequencies[i]
                if frequencies[i] > max_freq:
                    max_freq = frequencies[i]
        
        return float(min_freq), float(max_freq)
    
    def compute_magnitude_range(self, samples: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute min/max magnitude and their corresponding frequencies."""
        frequencies, magnitudes = self.compute_magnitude_spectrum(samples)
        
        if len(magnitudes) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        min_mag = magnitudes[1] if len(magnitudes) > 1 else magnitudes[0]
        max_mag = magnitudes[1] if len(magnitudes) > 1 else magnitudes[0]
        min_idx = 1 if len(magnitudes) > 1 else 0
        max_idx = 1 if len(magnitudes) > 1 else 0
        
        for i in range(1, len(magnitudes)):
            if magnitudes[i] < min_mag:
                min_mag = magnitudes[i]
                min_idx = i
            if magnitudes[i] > max_mag:
                max_mag = magnitudes[i]
                max_idx = i
        
        return (float(min_mag), float(max_mag),
                float(frequencies[min_idx]), float(frequencies[max_idx]))
    
    # =========================================================================
    # Analysis Pipeline
    # =========================================================================
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Run full spectral analysis on an audio file."""
        samples, sample_rate = self.load_audio(file_path)
        self.sample_rate = sample_rate
        
        min_freq, max_freq = self.compute_frequency_range(samples)
        min_mag, max_mag, freq_at_min, freq_at_max = self.compute_magnitude_range(samples)
        
        results = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'sample_rate': sample_rate,
            'n_fft': self.n_fft,
            'average_power': self.compute_average_power(samples),
            'min_frequency_hz': min_freq,
            'max_frequency_hz': max_freq,
            'min_magnitude': min_mag,
            'max_magnitude': max_mag,
            'frequency_at_min_magnitude': freq_at_min,
            'frequency_at_max_magnitude': freq_at_max,
            'dominant_frequency_hz': freq_at_max,
        }
        
        return results
    
    # =========================================================================
    # Print Details
    # =========================================================================
    
    def print_analysis_results(self, results: Dict[str, Any]) -> None:
        """Print analysis results in a formatted way."""
        if not self.print_details:
            return
        
        print(f"\n   Spectral Analysis: {results.get('file_name', 'Unknown')}")
        print(f"   {'-' * 50}")
        print(f"   Sample Rate: {results['sample_rate']} Hz | FFT Size: {results['n_fft']}")
        print(f"   Average Power: {results['average_power']:.6f}")
        print(f"   Frequency Range: [{results['min_frequency_hz']:.1f}, "
              f"{results['max_frequency_hz']:.1f}] Hz")
        print(f"   Dominant Frequency: {results['dominant_frequency_hz']:.1f} Hz")
        print(f"   Magnitude Range: [{results['min_magnitude']:.4f}, "
              f"{results['max_magnitude']:.4f}]")
    
    def print_batch_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Print summary statistics for batch analysis."""
        if not self.print_details or not all_results:
            return
        
        print(f"\n   Batch Summary ({len(all_results)} files)")
        print(f"   {'-' * 50}")
        
        metrics = ['average_power', 'dominant_frequency_hz', 'min_frequency_hz', 'max_frequency_hz']
        for metric in metrics:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"   {metric}: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def _ensure_output_dir(self, class_name: Optional[str] = None) -> Path:
        """Ensure output directory exists and return path."""
        if class_name:
            output_path = self.output_dir / class_name
        else:
            output_path = self.output_dir
        
        if self.save_plots:
            output_path.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def plot_magnitude_spectrum(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        file_id: Optional[str] = None,
        log_scale: bool = True
    ) -> None:
        """Plot the magnitude spectrum of an audio file."""
        samples, sample_rate = self.load_audio(file_path)
        self.sample_rate = sample_rate
        frequencies, magnitudes = self.compute_magnitude_spectrum(samples)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        if log_scale:
            eps = 1e-10
            magnitudes_db = 20 * np.log10(magnitudes + eps)
            ax.plot(frequencies, magnitudes_db, linewidth=0.8, color='steelblue')
            ax.set_ylabel('Magnitude (dB)')
        else:
            ax.plot(frequencies, magnitudes, linewidth=0.8, color='steelblue')
            ax.set_ylabel('Magnitude')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(f'Magnitude Spectrum: {Path(file_path).name}')
        ax.set_xlim(0, sample_rate / 2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_path = self._ensure_output_dir(class_name)
            if file_id:
                save_path = output_path / f'{file_id}_spectrum.png'
            else:
                save_path = output_path / 'spectrum.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_psd(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> None:
        """Plot Power Spectral Density."""
        samples, sample_rate = self.load_audio(file_path)
        self.sample_rate = sample_rate
        frequencies, psd = self.compute_power_spectral_density(samples)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        eps = 1e-10
        psd_db = 10 * np.log10(psd + eps)
        
        ax.plot(frequencies, psd_db, linewidth=0.8, color='darkgreen')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.set_title(f'Power Spectral Density: {Path(file_path).name}')
        ax.set_xlim(0, sample_rate / 2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_path = self._ensure_output_dir(class_name)
            if file_id:
                save_path = output_path / f'{file_id}_psd.png'
            else:
                save_path = output_path / 'psd.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_spectrogram(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        file_id: Optional[str] = None,
        log_scale: bool = True,
        cmap: str = 'viridis'
    ) -> None:
        """Plot spectrogram of an audio file."""
        samples, sample_rate = self.load_audio(file_path)
        self.sample_rate = sample_rate
        times, frequencies, spectrogram = self.compute_spectrogram(samples)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if log_scale:
            eps = 1e-10
            spectrogram_db = 20 * np.log10(spectrogram + eps)
            im = ax.pcolormesh(times, frequencies, spectrogram_db,
                               shading='gouraud', cmap=cmap)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Magnitude (dB)')
        else:
            im = ax.pcolormesh(times, frequencies, spectrogram,
                               shading='gouraud', cmap=cmap)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Magnitude')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram: {Path(file_path).name}')
        ax.set_ylim(0, sample_rate / 2)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_path = self._ensure_output_dir(class_name)
            if file_id:
                save_path = output_path / f'{file_id}_spectrogram.png'
            else:
                save_path = output_path / 'spectrogram.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    # =========================================================================
    # Main Run Method
    # =========================================================================
    
    def run(
        self,
        dataset: Optional["BirdSoundDataset"] = None,
        train_dataset: Optional["BirdSoundDataset"] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the pipeline based on configuration.
        
        Args:
            dataset: Full dataset (used if run_mode='dataset')
            train_dataset: Training dataset (used if run_mode='train')
            
        Returns:
            List of analysis results.
        """
        if not self.enabled:
            print("   Spectral Analysis Pipeline is disabled.")
            return []
        
        print("\n   Running Spectral Analysis Pipeline (Frequency-Domain)...")
        print(f"   Mode: {self.run_mode}")
        
        all_results = []
        
        if self.run_mode == 'single':
            if self.single_file:
                results = self.analyze(self.single_file)
                results['class_label'] = 'single'
                all_results.append(results)
                self.print_analysis_results(results)
                self._generate_all_plots(self.single_file, class_name='single')
            else:
                print("   Warning: No single_file specified in config.")
        
        elif self.run_mode == 'folder':
            if self.folder_path:
                folder = Path(self.folder_path)
                if folder.exists():
                    class_name = folder.name
                    audio_files = list(folder.glob('*.mp3')) + list(folder.glob('*.wav'))
                    
                    print(f"   Processing {len(audio_files)} files from {class_name}...")
                    
                    for audio_file in audio_files:
                        results = self.analyze(str(audio_file))
                        results['class_label'] = class_name
                        all_results.append(results)
                        
                        file_id = audio_file.stem
                        self._generate_all_plots(str(audio_file), class_name=class_name,
                                                file_id=file_id)
                    
                    self.print_batch_summary(all_results)
                else:
                    print(f"   Warning: Folder not found: {self.folder_path}")
            else:
                print("   Warning: No folder_path specified in config.")
        
        elif self.run_mode == 'train':
            if train_dataset and len(train_dataset) > 0:
                print(f"   Processing {len(train_dataset)} training samples...")
                
                for i in range(len(train_dataset)):
                    file_path, label_idx = train_dataset[i]
                    class_name = train_dataset.get_class_name(label_idx)
                    
                    results = self.analyze(file_path)
                    results['class_label'] = class_name
                    all_results.append(results)
                    
                    file_id = Path(file_path).stem
                    self._generate_all_plots(file_path, class_name=class_name, file_id=file_id)
                
                self.print_batch_summary(all_results)
            else:
                print("   Warning: No training dataset provided.")
        
        elif self.run_mode == 'dataset':
            if dataset and len(dataset) > 0:
                print(f"   Processing {len(dataset)} samples from full dataset...")
                
                for i in range(len(dataset)):
                    file_path, label_idx = dataset[i]
                    class_name = dataset.get_class_name(label_idx)
                    
                    results = self.analyze(file_path)
                    results['class_label'] = class_name
                    all_results.append(results)
                    
                    file_id = Path(file_path).stem
                    self._generate_all_plots(file_path, class_name=class_name, file_id=file_id)
                
                self.print_batch_summary(all_results)
            else:
                print("   Warning: No dataset provided.")
        
        else:
            print(f"   Warning: Unknown run_mode: {self.run_mode}")
        
        if self.save_plots and all_results:
            print(f"   Plots saved to: {self.output_dir}")
        
        return all_results
    
    def _generate_all_plots(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> None:
        """Generate all spectral plots for a file."""
        self.plot_magnitude_spectrum(file_path, class_name=class_name, file_id=file_id)
        self.plot_psd(file_path, class_name=class_name, file_id=file_id)
        self.plot_spectrogram(file_path, class_name=class_name, file_id=file_id)
