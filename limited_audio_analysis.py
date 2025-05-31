#!/usr/bin/env python3
"""
Limited Analysis Version for Wow! Signal Audio Analysis

This script is a memory-optimized version of the audio_analysis.py script,
intended for use when the original script causes system crashes due to memory constraints.

It performs only the most essential analyses while strictly limiting memory usage.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

class LimitedWowSignalAnalysis:
    """Memory-optimized class for analyzing audio representation of the Wow! signal"""
    
    def __init__(self, audio_path):
        """Initialize with the path to the audio file"""
        print(f"Loading audio file: {audio_path}")
        self.audio_path = audio_path
        
        # Use a lower sample rate to reduce memory usage
        try:
            # Load with downsampling and limited duration
            print("Loading audio with reduced sample rate...")
            self.y, self.sr = librosa.load(audio_path, sr=16000, mono=True, duration=120)  # 2 minutes max
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            print(f"Audio loaded successfully: {self.duration:.2f} seconds, {self.sr} Hz sample rate")
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
            
        # Create output directory
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.project_root, 'results', 'limited_audio_analysis')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_analysis(self):
        """Run a limited set of analyses to avoid memory issues"""
        print("\nRunning limited audio analysis...")
        
        # Configure matplotlib to use less memory
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['figure.figsize'] = (8, 4)
        
        # Store all results
        results = {}
        
        # Basic audio information
        results["basic_info"] = {
            "filename": os.path.basename(self.audio_path),
            "duration": self.duration,
            "sample_rate": self.sr,
            "num_samples": len(self.y)
        }
        
        # 1. Basic waveform visualization
        try:
            print("Generating basic waveform...")
            waveform_path = self.generate_waveform()
            results["waveform_path"] = waveform_path
        except Exception as e:
            print(f"Error generating waveform: {e}")
            results["waveform_path"] = None
        
        # 2. Simple spectrogram
        try:
            print("Generating simple spectrogram...")
            spectrogram_path = self.generate_simple_spectrogram()
            results["spectrogram_path"] = spectrogram_path
        except Exception as e:
            print(f"Error generating spectrogram: {e}")
            results["spectrogram_path"] = None
        
        # 3. Basic spectral analysis
        try:
            print("Performing basic spectral analysis...")
            spectral_info = self.basic_spectral_analysis()
            results["spectral_info"] = spectral_info
        except Exception as e:
            print(f"Error in spectral analysis: {e}")
            results["spectral_info"] = {"error": str(e)}
        
        # Generate simple summary
        try:
            print("Generating simple summary...")
            summary_path = self.generate_simple_summary(results)
            results["summary_path"] = summary_path
        except Exception as e:
            print(f"Error generating summary: {e}")
            results["summary_path"] = None
        
        print(f"Limited analysis complete. Results saved to {self.results_dir}")
        return results
    
    def generate_waveform(self):
        """Generate a simple waveform visualization"""
        # Downsample for visualization if needed
        max_points = 5000
        if len(self.y) > max_points:
            step = len(self.y) // max_points
            y_plot = self.y[::step]
            times = np.linspace(0, self.duration, len(y_plot))
        else:
            y_plot = self.y
            times = np.linspace(0, self.duration, len(self.y))
        
        plt.figure()
        plt.plot(times, y_plot)
        plt.title('Wow! Signal Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'waveform_simple.png')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def generate_simple_spectrogram(self):
        """Generate a basic spectrogram with memory constraints"""
        # Use a smaller FFT size to reduce memory usage
        n_fft = 1024
        hop_length = n_fft // 4
        
        # Only process first minute max if file is longer
        max_samples = 60 * self.sr
        if len(self.y) > max_samples:
            y_spec = self.y[:max_samples]
        else:
            y_spec = self.y
        
        plt.figure()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_spec, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        plt.imshow(D, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Wow! Signal Spectrogram (Simple)')
        plt.ylabel('Frequency Bin')
        plt.xlabel('Time Frame')
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'spectrogram_simple.png')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def basic_spectral_analysis(self):
        """Perform basic spectral analysis with minimal memory usage"""
        # Calculate basic spectral features
        # Use a reasonable FFT size to avoid memory issues
        n_fft = 2048
        
        # Calculate spectrum
        magnitude = np.abs(librosa.stft(self.y, n_fft=n_fft))
        magnitude_mean = np.mean(magnitude, axis=1)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # Find dominant frequencies (top 5)
        sorted_idx = np.argsort(magnitude_mean)[::-1][:5]
        dominant_freqs = freqs[sorted_idx]
        dominant_mags = magnitude_mean[sorted_idx]
        
        # Calculate basic spectral statistics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0])
        
        # Plot frequency spectrum (downsampled if needed)
        plt.figure()
        if len(magnitude_mean) > 1000:
            step = len(magnitude_mean) // 1000
            plt.semilogy(freqs[::step], magnitude_mean[::step])
        else:
            plt.semilogy(freqs, magnitude_mean)
            
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (log scale)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        spectrum_path = os.path.join(self.results_dir, 'spectrum_simple.png')
        plt.savefig(spectrum_path)
        plt.close()
        
        return {
            "spectrum_path": spectrum_path,
            "dominant_frequencies": dominant_freqs.tolist(),
            "dominant_magnitudes": dominant_mags.tolist(),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_rolloff": float(spectral_rolloff)
        }
    
    def generate_simple_summary(self, results):
        """Generate a simple summary of the limited analysis"""
        output_path = os.path.join(self.results_dir, 'limited_analysis_summary.md')
        
        with open(output_path, 'w') as f:
            f.write("# Wow! Signal Limited Audio Analysis\n\n")
            f.write("*This is a memory-optimized limited analysis report.*\n\n")
            
            f.write("## File Information\n\n")
            f.write(f"- **Filename:** {results['basic_info']['filename']}\n")
            f.write(f"- **Duration:** {results['basic_info']['duration']:.2f} seconds\n")
            f.write(f"- **Sample Rate:** {results['basic_info']['sample_rate']} Hz\n\n")
            
            f.write("## Visualizations\n\n")
            
            if results["waveform_path"]:
                f.write(f"![Waveform]({os.path.basename(results['waveform_path'])})\n\n")
            
            if results["spectrogram_path"]:
                f.write(f"![Spectrogram]({os.path.basename(results['spectrogram_path'])})\n\n")
            
            if "spectral_info" in results and "spectrum_path" in results["spectral_info"]:
                f.write(f"![Frequency Spectrum]({os.path.basename(results['spectral_info']['spectrum_path'])})\n\n")
            
            if "spectral_info" in results:
                f.write("## Spectral Analysis\n\n")
                
                if "spectral_centroid" in results["spectral_info"]:
                    f.write(f"- **Spectral Centroid:** {results['spectral_info']['spectral_centroid']:.2f} Hz\n")
                    f.write(f"- **Spectral Bandwidth:** {results['spectral_info']['spectral_bandwidth']:.2f} Hz\n")
                    f.write(f"- **Spectral Rolloff:** {results['spectral_info']['spectral_rolloff']:.2f} Hz\n\n")
                
                if "dominant_frequencies" in results["spectral_info"]:
                    f.write("### Dominant Frequencies\n\n")
                    f.write("| Frequency (Hz) | Magnitude |\n")
                    f.write("|---------------|----------|\n")
                    
                    for freq, mag in zip(results["spectral_info"]["dominant_frequencies"], 
                                        results["spectral_info"]["dominant_magnitudes"]):
                        f.write(f"| {freq:.2f} | {mag:.4f} |\n")
                    f.write("\n")
            
            f.write("## Notes\n\n")
            f.write("This is a limited analysis performed with strict memory constraints to avoid system crashes. ")
            f.write("For a more comprehensive analysis, consider processing the file on a system with more memory ")
            f.write("or reducing the audio file's size/duration/sample rate before analysis.\n\n")
            
            f.write("---\n\n")
            f.write("*Analysis completed on a memory-constrained system.*\n")
        
        return output_path


def main():
    """Main entry point for the script"""
    print("\n" + "="*80)
    print(" "*30 + "WOW! SIGNAL LIMITED ANALYSIS")
    print("="*80 + "\n")
    
    # Path to the audio file
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'data', 'Wow_Signal_SETI_Project.mp3')
    
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found at {audio_path}")
            return 1
            
        # Use non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')
        
        # Create analyzer object
        analyzer = LimitedWowSignalAnalysis(audio_path)
        
        # Run the limited analysis
        results = analyzer.run_analysis()
        
        print("\n" + "="*80)
        print(f"Limited analysis complete! Results saved to {analyzer.results_dir}")
        print("="*80 + "\n")
        
    except MemoryError:
        print("\nERROR: Out of memory even with the limited analysis.")
        print("Try using a smaller audio file or a system with more RAM.")
        return 1
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()
