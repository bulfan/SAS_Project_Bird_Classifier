"""
Data Exploration Pipeline for Bird Sound Classification.

Time-domain statistical analysis of audio signals.
Implements all algorithms from scratch (no direct library implementations).

Metrics computed:
- Root Mean Square (RMS)
- Standard Deviation
- Crest Factor
- Average Amplitude
- Min/Max Amplitude
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.data.dataset import BirdSoundDataset


class DataExplorationPipeline:
    """
    Pipeline for time-domain data exploration of audio signals.
    
    All statistical methods are implemented from scratch without using
    direct library implementations (e.g., np.std is not used).
    """
    
    def __init__(self, cfg: Optional[Any] = None):
        """
        Initialize the Data Exploration Pipeline.
        
        Args:
            cfg: Hydra configuration object or None for defaults.
        """
        if cfg is not None:
            exploration_cfg = cfg.get('analysis', {}).get('exploration', {})
            self.frame_ms = exploration_cfg.get('frame_ms', None)
            self.output_dir = Path(exploration_cfg.get('output_dir', 'outputs/time'))
            self.save_plots = exploration_cfg.get('save_plots', True)
            self.show_plots = exploration_cfg.get('show_plots', False)
            self.print_details = exploration_cfg.get('print_details', True)
            self.enabled = exploration_cfg.get('enabled', True)
            self.run_mode = exploration_cfg.get('run_mode', 'single')
            self.single_file = exploration_cfg.get('single_file', None)
            self.folder_path = exploration_cfg.get('folder_path', None)
            self.sample_rate = cfg.get('data', {}).get('sample_rate', 22050)
        else:
            self.frame_ms = None
            self.output_dir = Path('outputs/time')
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
    # Time-Domain Statistical Metrics (Implemented from Scratch)
    # =========================================================================
    
    def compute_rms(self, samples: np.ndarray) -> float:
        """Compute Root Mean Square (RMS) of the signal."""
        n = len(samples)
        if n == 0:
            return 0.0
        
        sum_of_squares = 0.0
        for i in range(n):
            sum_of_squares += samples[i] * samples[i]
        
        mean_of_squares = sum_of_squares / n
        rms = np.sqrt(mean_of_squares)
        
        return float(rms)
    
    def compute_standard_deviation(self, samples: np.ndarray) -> float:
        """Compute Standard Deviation of the signal."""
        n = len(samples)
        if n == 0:
            return 0.0
        
        total = 0.0
        for i in range(n):
            total += samples[i]
        mean = total / n
        
        sum_squared_dev = 0.0
        for i in range(n):
            deviation = samples[i] - mean
            sum_squared_dev += deviation * deviation
        
        variance = sum_squared_dev / n
        std = np.sqrt(variance)
        
        return float(std)
    
    def compute_average_amplitude(self, samples: np.ndarray) -> float:
        """Compute Average Amplitude (mean of absolute values)."""
        n = len(samples)
        if n == 0:
            return 0.0
        
        sum_abs = 0.0
        for i in range(n):
            if samples[i] >= 0:
                sum_abs += samples[i]
            else:
                sum_abs += -samples[i]
        
        return float(sum_abs / n)
    
    def compute_min_max_amplitude(self, samples: np.ndarray) -> Tuple[float, float]:
        """Compute Min and Max amplitude of the signal."""
        if len(samples) == 0:
            return 0.0, 0.0
        
        min_val = samples[0]
        max_val = samples[0]
        
        for i in range(1, len(samples)):
            if samples[i] < min_val:
                min_val = samples[i]
            if samples[i] > max_val:
                max_val = samples[i]
        
        return float(min_val), float(max_val)
    
    def compute_crest_factor(self, samples: np.ndarray) -> float:
        """Compute Crest Factor of the signal."""
        if len(samples) == 0:
            return 0.0
        
        peak = 0.0
        for i in range(len(samples)):
            abs_val = samples[i] if samples[i] >= 0 else -samples[i]
            if abs_val > peak:
                peak = abs_val
        
        rms = self.compute_rms(samples)
        
        if rms == 0:
            return 0.0
        
        return float(peak / rms)
    
    # =========================================================================
    # Analysis Pipeline
    # =========================================================================
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Run full time-domain analysis on an audio file."""
        samples, sample_rate = self.load_audio(file_path)
        
        min_amp, max_amp = self.compute_min_max_amplitude(samples)
        
        results = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'sample_rate': sample_rate,
            'num_samples': len(samples),
            'duration_seconds': len(samples) / sample_rate,
            'rms': self.compute_rms(samples),
            'standard_deviation': self.compute_standard_deviation(samples),
            'average_amplitude': self.compute_average_amplitude(samples),
            'min_amplitude': min_amp,
            'max_amplitude': max_amp,
            'crest_factor': self.compute_crest_factor(samples),
        }
        
        return results
    
    # =========================================================================
    # Print Details
    # =========================================================================
    
    def print_analysis_results(self, results: Dict[str, Any]) -> None:
        """Print analysis results in a formatted way."""
        if not self.print_details:
            return
        
        print(f"\n   Time-Domain Analysis: {results.get('file_name', 'Unknown')}")
        print(f"   {'-' * 50}")
        print(f"   Duration: {results['duration_seconds']:.2f}s | "
              f"Sample Rate: {results['sample_rate']} Hz")
        print(f"   RMS: {results['rms']:.6f}")
        print(f"   Standard Deviation: {results['standard_deviation']:.6f}")
        print(f"   Crest Factor: {results['crest_factor']:.2f}")
        print(f"   Average Amplitude: {results['average_amplitude']:.6f}")
        print(f"   Min/Max Amplitude: [{results['min_amplitude']:.4f}, "
              f"{results['max_amplitude']:.4f}]")
    
    def print_batch_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Print summary statistics for batch analysis."""
        if not self.print_details or not all_results:
            return
        
        print(f"\n   Batch Summary ({len(all_results)} files)")
        print(f"   {'-' * 50}")
        
        metrics = ['rms', 'standard_deviation', 'crest_factor', 'average_amplitude']
        for metric in metrics:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"   {metric}: mean={mean_val:.4f}, "
                      f"min={min_val:.4f}, max={max_val:.4f}")
    
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
    
    def plot_waveform(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> None:
        """Plot the waveform of an audio file."""
        samples, sample_rate = self.load_audio(file_path)
        
        duration = len(samples) / sample_rate
        time_axis = np.linspace(0, duration, len(samples))
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, samples, linewidth=0.5, color='steelblue')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform: {Path(file_path).name}')
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_path = self._ensure_output_dir(class_name)
            if file_id:
                save_path = output_path / f'{file_id}_waveform.png'
            else:
                save_path = output_path / 'waveform.png'
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
            print("   Data Exploration Pipeline is disabled.")
            return []
        
        print("\n   Running Data Exploration Pipeline (Time-Domain)...")
        print(f"   Mode: {self.run_mode}")
        
        all_results = []
        
        if self.run_mode == 'single':
            # Single file mode
            if self.single_file:
                results = self.analyze(self.single_file)
                results['class_label'] = 'single'
                all_results.append(results)
                self.print_analysis_results(results)
                self.plot_waveform(self.single_file, class_name='single')
            else:
                print("   Warning: No single_file specified in config.")
        
        elif self.run_mode == 'folder':
            # Folder mode
            if self.folder_path:
                folder = Path(self.folder_path)
                if folder.exists():
                    class_name = folder.name
                    audio_files = list(folder.glob('*.mp3')) + list(folder.glob('*.wav'))
                    
                    print(f"   Processing {len(audio_files)} files from {class_name}...")
                    
                    for i, audio_file in enumerate(audio_files):
                        results = self.analyze(str(audio_file))
                        results['class_label'] = class_name
                        all_results.append(results)
                        
                        file_id = audio_file.stem
                        self.plot_waveform(str(audio_file), class_name=class_name, 
                                          file_id=file_id)
                    
                    self.print_batch_summary(all_results)
                else:
                    print(f"   Warning: Folder not found: {self.folder_path}")
            else:
                print("   Warning: No folder_path specified in config.")
        
        elif self.run_mode == 'train':
            # Training dataset mode
            if train_dataset and len(train_dataset) > 0:
                print(f"   Processing {len(train_dataset)} training samples...")
                
                for i in range(len(train_dataset)):
                    file_path, label_idx = train_dataset[i]
                    class_name = train_dataset.get_class_name(label_idx)
                    
                    results = self.analyze(file_path)
                    results['class_label'] = class_name
                    all_results.append(results)
                    
                    file_id = Path(file_path).stem
                    self.plot_waveform(file_path, class_name=class_name, file_id=file_id)
                
                self.print_batch_summary(all_results)
            else:
                print("   Warning: No training dataset provided.")
        
        elif self.run_mode == 'dataset':
            # Full dataset mode
            if dataset and len(dataset) > 0:
                print(f"   Processing {len(dataset)} samples from full dataset...")
                
                for i in range(len(dataset)):
                    file_path, label_idx = dataset[i]
                    class_name = dataset.get_class_name(label_idx)
                    
                    results = self.analyze(file_path)
                    results['class_label'] = class_name
                    all_results.append(results)
                    
                    file_id = Path(file_path).stem
                    self.plot_waveform(file_path, class_name=class_name, file_id=file_id)
                
                self.print_batch_summary(all_results)
            else:
                print("   Warning: No dataset provided.")
        
        else:
            print(f"   Warning: Unknown run_mode: {self.run_mode}")
        
        if self.save_plots and all_results:
            print(f"   Plots saved to: {self.output_dir}")
        
        return all_results
