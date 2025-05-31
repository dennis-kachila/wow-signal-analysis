#!/usr/bin/env python3
"""
Wow! Signal Audio Analysis

This script analyzes the audio representation of the Wow! signal,
performing spectral analysis, visualization, and feature extraction
to identify patterns and characteristics that might provide insight
into the signal's origins.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
from scipy.io import wavfile
import pywt
from tqdm import tqdm
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

class WowSignalAudioAnalysis:
    """Class for analyzing audio representation of the Wow! signal"""
    
    def __init__(self, audio_path, sr=22050, max_duration=None):
        """
        Initialize with the path to the audio file
        
        Args:
            audio_path: Path to the audio file
            sr: Target sample rate for loading audio (downsampling if needed)
            max_duration: Maximum duration in seconds to load (None for full file)
        """
        print(f"Loading audio file: {audio_path}")
        self.audio_path = audio_path
        
        # Load the audio file with memory optimizations
        try:
            # Use a fixed sample rate to reduce memory usage
            # Default sr=22050 should be sufficient for most analysis tasks
            if max_duration:
                print(f"Loading only first {max_duration} seconds of audio...")
                self.y, self.sr = librosa.load(audio_path, sr=sr, duration=max_duration)
            else:
                self.y, self.sr = librosa.load(audio_path, sr=sr)
            
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            print(f"Audio loaded successfully: {self.duration:.2f} seconds, {self.sr} Hz sample rate")
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
            
        # Create output directory
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.project_root, 'results', 'audio_analysis')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_full_analysis(self):
        """Run all analysis methods and compile results"""
        print("\nRunning full audio analysis...")
        
        results = {
            "file_info": {
                "filename": os.path.basename(self.audio_path),
                "duration": self.duration,
                "sample_rate": self.sr,
                "num_samples": len(self.y)
            }
        }
        
        # Use try/except for each step to prevent a single failure from stopping the analysis
        try:
            # Waveform visualization
            print("Generating waveform visualization...")
            waveform_path = self.visualize_waveform()
            results["waveform_path"] = waveform_path
        except Exception as e:
            print(f"WARNING: Error during waveform visualization: {e}")
            results["waveform_path"] = None
        
        # Clear memory between steps
        import gc
        gc.collect()
        
        try:
            # Spectral analysis
            print("Performing spectral analysis...")
            spectral_results = self.analyze_spectrum()
            results["spectral_analysis"] = spectral_results
        except Exception as e:
            print(f"WARNING: Error during spectral analysis: {e}")
            results["spectral_analysis"] = {"error": str(e)}
            
        gc.collect()
        
        try:
            # Spectrogram analysis
            print("Generating spectrogram analysis...")
            spectrogram_path = self.generate_spectrogram()
            results["spectrogram_path"] = spectrogram_path
        except Exception as e:
            print(f"WARNING: Error generating spectrogram: {e}")
            results["spectrogram_path"] = None
            
        gc.collect()
        
        try:
            # Mel spectrogram
            print("Generating mel spectrogram...")
            mel_spectrogram_path = self.generate_mel_spectrogram()
            results["mel_spectrogram_path"] = mel_spectrogram_path
        except Exception as e:
            print(f"WARNING: Error generating mel spectrogram: {e}")
            results["mel_spectrogram_path"] = None
            
        gc.collect()
        
        try:
            # Chromagram
            print("Generating chromagram...")
            chromagram_path = self.generate_chromagram()
            results["chromagram_path"] = chromagram_path
        except Exception as e:
            print(f"WARNING: Error generating chromagram: {e}")
            results["chromagram_path"] = None
            
        gc.collect()
        
        try:
            # Onset detection
            print("Detecting onsets in signal...")
            onset_results = self.detect_onsets()
            results["onset_analysis"] = onset_results
        except Exception as e:
            print(f"WARNING: Error during onset detection: {e}")
            results["onset_analysis"] = {"error": str(e)}
            
        gc.collect()
        
        try:
            # Wavelet analysis
            print("Performing wavelet decomposition...")
            wavelet_path = self.perform_wavelet_analysis()
            results["wavelet_path"] = wavelet_path
        except Exception as e:
            print(f"WARNING: Error during wavelet analysis: {e}")
            results["wavelet_path"] = None
            
        gc.collect()
        
        try:
            # Pattern detection
            print("Detecting patterns in audio...")
            pattern_results = self.detect_patterns()
            results["pattern_analysis"] = pattern_results
        except Exception as e:
            print(f"WARNING: Error during pattern detection: {e}")
            results["pattern_analysis"] = {"error": str(e)}
            
        gc.collect()
        
        try:
            # Statistical analysis
            print("Performing statistical analysis...")
            stats_results = self.analyze_statistics()
            results["statistical_analysis"] = stats_results
        except Exception as e:
            print(f"WARNING: Error during statistical analysis: {e}")
            results["statistical_analysis"] = {"error": str(e)}
            
        gc.collect()
        
        try:
            # Modulation analysis
            print("Analyzing modulation characteristics...")
            mod_results = self.analyze_modulation()
            results["modulation_analysis"] = mod_results
        except Exception as e:
            print(f"WARNING: Error during modulation analysis: {e}")
            results["modulation_analysis"] = {"error": str(e)}
            
        gc.collect()
        
        # Generate summary
        try:
            print("Generating analysis summary...")
            summary_path = self.generate_summary_report(results)
            results["summary_path"] = summary_path
            
            print(f"\nAnalysis complete! Summary report saved to: {summary_path}")
        except Exception as e:
            print(f"WARNING: Error generating summary report: {e}")
            results["summary_path"] = None
        
        return results
    
    def visualize_waveform(self):
        """
        Visualize the audio waveform
        
        Returns:
            Path to the saved waveform plot
        """
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.8)
        plt.title('Wow! Signal Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_waveform.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def analyze_spectrum(self):
        """
        Analyze the frequency spectrum of the audio
        
        Returns:
            Dictionary with spectral analysis results and plot path
        """
        # Compute the FFT
        n_fft = 4096
        fft = np.abs(librosa.stft(self.y, n_fft=n_fft))
        magnitude = np.mean(fft, axis=1)
        frequency = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(magnitude, height=np.mean(magnitude)*1.5, distance=20)
        peak_freqs = frequency[peaks]
        peak_mags = magnitude[peaks]
        
        # Sort peaks by magnitude
        peak_idx = np.argsort(peak_mags)[::-1][:10]  # Top 10 peaks
        top_peaks = [(peak_freqs[i], peak_mags[i]) for i in peak_idx]
        
        # Plot the spectrum
        plt.figure(figsize=(12, 6))
        plt.semilogy(frequency, magnitude)
        plt.plot(peak_freqs[peak_idx], peak_mags[peak_idx], 'ro', markersize=5)
        
        # Annotate only top 3 peaks to reduce clutter
        for i, (freq, mag) in enumerate(top_peaks[:3]):
            plt.annotate(f"{freq:.1f} Hz", (freq, mag), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, color='red')
            
        plt.title('Wow! Signal Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (log scale)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_spectrum.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # Calculate spectral centroid
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        mean_centroid = np.mean(centroid)
        
        # Calculate spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
        mean_bandwidth = np.mean(bandwidth)
        
        # Calculate spectral contrast
        contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        mean_contrast = np.mean(contrast, axis=1)
        
        # Calculate spectral flatness
        flatness = librosa.feature.spectral_flatness(y=self.y)[0]
        mean_flatness = np.mean(flatness)
        
        # Calculate spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        mean_rolloff = np.mean(rolloff)
        
        return {
            "spectrum_plot": output_path,
            "top_peaks": top_peaks,
            "spectral_centroid": float(mean_centroid),
            "spectral_bandwidth": float(mean_bandwidth),
            "spectral_contrast": mean_contrast.tolist(),
            "spectral_flatness": float(mean_flatness),
            "spectral_rolloff": float(mean_rolloff)
        }
    
    def generate_spectrogram(self):
        """
        Generate a spectrogram visualization
        
        Returns:
            Path to the saved spectrogram
        """
        plt.figure(figsize=(12, 8))
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Wow! Signal Spectrogram')
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_spectrogram.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def generate_mel_spectrogram(self):
        """
        Generate a mel spectrogram visualization
        
        Returns:
            Path to the saved mel spectrogram
        """
        plt.figure(figsize=(12, 8))
        
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=self.sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Wow! Signal Mel Spectrogram')
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_mel_spectrogram.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def generate_chromagram(self):
        """
        Generate a chromagram visualization
        
        Returns:
            Path to the saved chromagram
        """
        plt.figure(figsize=(12, 6))
        
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, n_chroma=12)
        librosa.display.specshow(chroma, sr=self.sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
        plt.colorbar()
        plt.title('Wow! Signal Chromagram')
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_chromagram.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def detect_onsets(self):
        """
        Detect onsets in the audio signal
        
        Returns:
            Dictionary with onset detection results and plot path
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Detect onsets
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
        onset_times = librosa.frames_to_time(onsets, sr=self.sr)
        
        # Compute onset density over time
        frame_length = int(self.sr * 0.1)  # 100ms frames
        hop_length = int(frame_length / 2)
        n_frames = 1 + (len(self.y) - frame_length) // hop_length
        onset_density = np.zeros(n_frames)
        
        for i in range(n_frames):
            start_time = i * hop_length / self.sr
            end_time = (i * hop_length + frame_length) / self.sr
            onset_density[i] = np.sum((onset_times >= start_time) & (onset_times < end_time))
        
        # Plot onsets - use downsampling for waveform plotting if needed
        plt.figure(figsize=(10, 6))
        
        # Plot waveform (downsampled for large files)
        plt.subplot(2, 1, 1)
        max_plot_points = 10000  # Limit to prevent excessive memory use
        if len(self.y) > max_plot_points:
            step = len(self.y) // max_plot_points
            y_plot = self.y[::step]
            times = np.arange(0, len(self.y), step) / self.sr
        else:
            y_plot = self.y
            times = np.arange(len(self.y)) / self.sr
        
        plt.plot(times, y_plot, alpha=0.5)
        plt.vlines(onset_times, -1, 1, color='r', alpha=0.9)
        plt.title('Detected Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot onset strength
        plt.subplot(2, 1, 2)
        frames = np.arange(len(onset_env))
        times = librosa.frames_to_time(frames, sr=self.sr)
        plt.plot(times, onset_env)
        plt.vlines(onset_times, 0, onset_env.max(), color='r', alpha=0.9)
        plt.title('Onset Strength')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_onsets.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return {
            "onset_plot": output_path,
            "num_onsets": len(onsets),
            "onset_times": onset_times.tolist(),
            "onset_density": np.mean(onset_density),
            "onset_regularity": np.std(np.diff(onset_times)) if len(onset_times) > 1 else None
        }
    
    def perform_wavelet_analysis(self):
        """
        Perform wavelet decomposition of the signal
        
        Returns:
            Path to the saved wavelet plot
        """
        # Downsample the signal to reduce memory usage
        # Take at most 10 seconds of audio, downsampled to reduce memory usage
        max_samples = 10 * self.sr  # 10 seconds max
        step = max(1, len(self.y) // max_samples)
        y_downsampled = self.y[::step]
        
        # Use fewer scales for the wavelet transform to reduce memory usage
        max_scale = min(64, len(y_downsampled) // 10)  # Limit scale based on signal length
        scales = np.arange(1, max_scale)
        
        try:
            # Compute the continuous wavelet transform with memory limits
            coeffs, freqs = pywt.cwt(y_downsampled, scales, 'morl')
            
            plt.figure(figsize=(10, 6))
            plt.imshow(np.abs(coeffs), aspect='auto', interpolation='nearest', cmap='viridis',
                     extent=[0, len(y_downsampled) / self.sr * step, 1, max_scale])
            plt.colorbar(label='Magnitude')
            plt.ylabel('Scale')
            plt.xlabel('Time (s)')
            plt.title('Wow! Signal Wavelet Transform (Downsampled)')
            plt.tight_layout()
            
            output_path = os.path.join(self.results_dir, 'wow_signal_wavelet.png')
            plt.savefig(output_path, dpi=200)  # Lower DPI to reduce memory usage
            plt.close()
            
            return output_path
            
        except MemoryError:
            print("WARNING: Memory error during wavelet analysis. Skipping this step.")
            # Create a simple placeholder image
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Wavelet analysis skipped - insufficient memory", 
                   ha='center', va='center', fontsize=14)
            plt.axis('off')
            
            output_path = os.path.join(self.results_dir, 'wow_signal_wavelet.png')
            plt.savefig(output_path, dpi=100)
            plt.close()
            
            return output_path
    
    def detect_patterns(self):
        """
        Detect patterns in the audio signal
        
        Returns:
            Dictionary with pattern detection results
        """
        # If signal is too long, analyze only a portion to save memory
        max_samples = 500000  # Maximum number of samples to use for autocorrelation
        
        if len(self.y) > max_samples:
            print(f"Signal too long for full autocorrelation analysis. Using first {max_samples/self.sr:.1f} seconds.")
            y_autocorr = self.y[:max_samples]
        else:
            y_autocorr = self.y
            
        # Calculate autocorrelation to find repeating patterns
        try:
            autocorr = librosa.autocorrelate(y_autocorr)
            
            # Normalize and take the second half (positive lags)
            autocorr = autocorr / (np.max(np.abs(autocorr)) + 1e-10)  # Avoid division by zero
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation (indicates possible periodicity)
            peaks, _ = signal.find_peaks(autocorr, height=0.2, distance=self.sr//50)
            peak_lags = peaks
            peak_times = peak_lags / self.sr
        except MemoryError:
            print("WARNING: Memory error during autocorrelation. Using simplified approach.")
            # Use a simplified approach with downsampling
            downsample_factor = len(self.y) // (max_samples//10) + 1
            y_simple = self.y[::downsample_factor]
            
            # Simple correlation calculation
            y_simple = y_simple - np.mean(y_simple)  # Remove DC
            autocorr = np.correlate(y_simple, y_simple, mode='full')
            autocorr = autocorr[len(autocorr)//2:] / np.max(autocorr)
            
            # Find major peaks
            peaks, _ = signal.find_peaks(autocorr, height=0.3)
            peak_lags = peaks
            peak_times = peak_lags / (self.sr/downsample_factor)
        
        # Plot autocorrelation
        plt.figure(figsize=(12, 6))
        lags = np.arange(len(autocorr))
        times = lags / self.sr
        plt.plot(times, autocorr)
        
        if len(peak_lags) > 0:
            peak_heights = autocorr[peak_lags]
            plt.plot(peak_times, peak_heights, 'ro')
            
            # Annotate the top 5 peaks
            sorted_idx = np.argsort(peak_heights)[::-1][:5]
            for i in sorted_idx:
                plt.annotate(f"{peak_times[i]:.3f}s", 
                           (peak_times[i], peak_heights[i]),
                           xytext=(5, 10), textcoords='offset points')
        
        plt.title('Autocorrelation - Potential Repeating Patterns')
        plt.xlabel('Lag (seconds)')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_autocorrelation.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # Estimate rhythmic periodicity if peaks found
        periodicity = None
        if len(peak_times) > 0:
            # Take the first significant peak
            first_peak = peak_times[0]
            periodicity = first_peak
        
        # Check for potential patterns in the frequency domain
        D = np.abs(librosa.stft(self.y))
        freq_corr = np.mean(np.corrcoef(D), axis=0)
        freq_pattern_strength = np.max(freq_corr)
        
        return {
            "autocorrelation_plot": output_path,
            "pattern_peaks": peak_times.tolist() if len(peak_times) > 0 else [],
            "estimated_periodicity": periodicity,
            "frequency_pattern_strength": float(freq_pattern_strength),
            "has_significant_patterns": freq_pattern_strength > 0.7 or (len(peak_times) > 0 and np.max(autocorr[peak_lags]) > 0.5)
        }
    
    def analyze_statistics(self):
        """
        Perform statistical analysis of the audio signal
        
        Returns:
            Dictionary with statistical analysis results
        """
        # Basic statistics
        mean = np.mean(self.y)
        std_dev = np.std(self.y)
        rms = np.sqrt(np.mean(self.y**2))
        peak = np.max(np.abs(self.y))
        dynamic_range = 20 * np.log10(peak / (np.mean(np.abs(self.y)) + 1e-10))
        
        # Skewness and kurtosis
        skewness = stats.skew(self.y)
        kurtosis = stats.kurtosis(self.y)
        
        # Zero-crossing rate
        zero_crossings = librosa.feature.zero_crossing_rate(self.y)[0]
        mean_zcr = np.mean(zero_crossings)
        
        # Spectrum statistics
        S = np.abs(librosa.stft(self.y))
        spectral_entropy = -np.sum(S * np.log2(S + 1e-10)) / S.size
        
        # Temporal statistics - divide into segments and analyze variation
        segment_length = self.sr  # 1 second segments
        n_segments = int(len(self.y) / segment_length)
        segment_means = []
        segment_stds = []
        
        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = self.y[start:end]
            segment_means.append(np.mean(segment))
            segment_stds.append(np.std(segment))
        
        temporal_variation = np.std(segment_means) if segment_means else 0
        
        # Plot statistical insights
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Amplitude distribution
        plt.subplot(2, 2, 1)
        plt.hist(self.y, bins=100, alpha=0.7)
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.3f}')
        plt.axvline(mean + std_dev, color='g', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev:.3f}')
        plt.axvline(mean - std_dev, color='g', linestyle='dashed', linewidth=1)
        plt.title('Amplitude Distribution')
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 2: Zero-crossing rate over time
        plt.subplot(2, 2, 2)
        zcr_times = librosa.frames_to_time(np.arange(len(zero_crossings)), sr=self.sr)
        plt.plot(zcr_times, zero_crossings)
        plt.axhline(mean_zcr, color='r', linestyle='dashed', linewidth=1, label=f'Mean ZCR: {mean_zcr:.3f}')
        plt.title('Zero-Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('ZCR')
        plt.legend()
        
        # Plot 3: Segment means over time
        plt.subplot(2, 2, 3)
        segment_times = np.arange(n_segments) * segment_length / self.sr
        plt.plot(segment_times, segment_means)
        plt.title('Segment Mean Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Amplitude')
        
        # Plot 4: Segment std devs over time
        plt.subplot(2, 2, 4)
        plt.plot(segment_times, segment_stds)
        plt.title('Segment Standard Deviation')
        plt.xlabel('Time (s)')
        plt.ylabel('Std Dev')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_statistics.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return {
            "statistics_plot": output_path,
            "mean_amplitude": float(mean),
            "std_dev": float(std_dev),
            "rms": float(rms),
            "peak_amplitude": float(peak),
            "dynamic_range_db": float(dynamic_range),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "mean_zero_crossing_rate": float(mean_zcr),
            "spectral_entropy": float(spectral_entropy),
            "temporal_variation": float(temporal_variation)
        }
    
    def analyze_modulation(self):
        """
        Analyze modulation characteristics of the signal
        
        Returns:
            Dictionary with modulation analysis results
        """
        # Reduce data for analysis - downsample if too large
        max_samples = 100000  # Reduce limit further to prevent memory issues
        if len(self.y) > max_samples:
            step = len(self.y) // max_samples
            y_analysis = self.y[::step]
            sr_analysis = self.sr / step
        else:
            y_analysis = self.y
            sr_analysis = self.sr
            
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        
        try:
            # Compute the amplitude envelope - ensure it's the same length as y_analysis
            envelope = np.abs(signal.hilbert(y_analysis))
            
            # Compute the instantaneous frequency
            analytic_signal = signal.hilbert(y_analysis)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * sr_analysis
            instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])  # Pad to match length
            
            # Remove outliers from instantaneous_frequency
            mean_freq = np.median(instantaneous_frequency)  # Use median instead of mean to be robust against outliers
            std_freq = np.std(instantaneous_frequency)
            mask = np.abs(instantaneous_frequency - mean_freq) < 5 * std_freq
            instantaneous_frequency_filtered = instantaneous_frequency[mask]
            
            # If filtered array is empty, use original
            if len(instantaneous_frequency_filtered) == 0:
                instantaneous_frequency_filtered = instantaneous_frequency
        
        except (MemoryError, ValueError) as e:
            print(f"WARNING: Error during signal processing: {e}. Using simplified analysis.")
            # Create dummy data for the plot
            envelope = y_analysis
            instantaneous_frequency = np.zeros_like(y_analysis)
            instantaneous_frequency_filtered = instantaneous_frequency
        
        # Calculate modulation frequency of the envelope - keep lengths consistent
        try:
            # Use a smaller portion for FFT if envelope is too large
            max_fft_size = 10000
            if len(envelope) > max_fft_size:
                env_for_fft = envelope[:max_fft_size]
            else:
                env_for_fft = envelope
                
            env_fft = np.abs(np.fft.fft(env_for_fft - np.mean(env_for_fft)))
            env_freqs = np.fft.fftfreq(len(env_for_fft), 1/sr_analysis)
            env_fft = env_fft[:len(env_fft)//2]
            env_freqs = env_freqs[:len(env_freqs)//2]
            env_freqs = env_freqs[env_freqs > 0]  # Remove DC component
            
            if len(env_freqs) > 0:  # Check if we have positive frequencies
                env_fft = env_fft[1:len(env_freqs)+1]
            else:
                env_fft = np.array([0])
                env_freqs = np.array([1])  # Dummy value
            
            # Find peaks in the envelope FFT
            if len(env_fft) > 0:
                peaks, _ = signal.find_peaks(env_fft, height=np.mean(env_fft)*2)
            else:
                peaks = []
            
            # If peaks found, get the dominant modulation frequency
            am_mod_freq = None
            if len(peaks) > 0 and len(env_freqs) > max(peaks) if peaks.size > 0 else 0:
                dominant_idx = np.argmax(env_fft[peaks])
                if dominant_idx < len(peaks):
                    peak_idx = peaks[dominant_idx]
                    if peak_idx < len(env_freqs):
                        am_mod_freq = env_freqs[peak_idx]
        except Exception as e:
            print(f"WARNING: Error during modulation frequency analysis: {e}")
            env_fft = np.array([0])
            env_freqs = np.array([1])
            peaks = []
            am_mod_freq = None
        
        # Calculate frequency modulation characteristics safely
        try:
            mean_freq = np.mean(instantaneous_frequency_filtered)
            if mean_freq > epsilon:
                fm_mod_depth = np.std(instantaneous_frequency_filtered) / mean_freq
            else:
                fm_mod_depth = 0
            
            # Modulation type detection (AM vs FM)
            mean_env = np.mean(envelope)
            if mean_env > epsilon:
                am_strength = np.var(envelope) / (mean_env**2)
            else:
                am_strength = 0
                
            mean_abs_freq = np.mean(np.abs(instantaneous_frequency_filtered))
            if mean_abs_freq > epsilon:
                fm_strength = np.var(instantaneous_frequency_filtered) / (mean_abs_freq**2) 
            else:
                fm_strength = 0
                
            # Determine modulation type
            mod_type = "Mixed"
            if am_strength > 2*fm_strength:
                mod_type = "Predominantly AM"
            elif fm_strength > 2*am_strength:
                mod_type = "Predominantly FM"
                
        except Exception as e:
            print(f"WARNING: Error during modulation strength calculation: {e}")
            # Default values
            fm_mod_depth = 0
            am_strength = 0
            fm_strength = 0
            mod_type = "Unknown"
        
        # Plot modulation characteristics
        plt.figure(figsize=(12, 10))
        
        try:
            # Plot 1: Signal and Envelope (downsampled for display)
            plt.subplot(3, 1, 1)
            # Downsample for plotting if needed
            max_plot_points = 5000  # Limit points to avoid memory issues
            if len(y_analysis) > max_plot_points:
                step = len(y_analysis) // max_plot_points
                times = np.arange(0, len(y_analysis), step) / sr_analysis
                y_plot = y_analysis[::step]
                
                # Make sure envelope has enough elements to downsample
                if len(envelope) >= len(y_analysis):
                    env_plot = envelope[::step]
                else:
                    # Create a compatible envelope
                    env_plot = np.zeros_like(y_plot)
            else:
                times = np.arange(len(y_analysis)) / sr_analysis
                y_plot = y_analysis
                
                # Ensure compatible lengths
                if len(envelope) == len(y_analysis):
                    env_plot = envelope
                else:
                    env_plot = np.zeros_like(y_plot)
                
            plt.plot(times, y_plot, alpha=0.5, label='Signal')
            plt.plot(times, env_plot, 'r', label='Envelope')
            plt.title('Signal and Amplitude Envelope')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            
            # Plot 2: Instantaneous Frequency
            plt.subplot(3, 1, 2)
            # Ensure the instantaneous_frequency is compatible with times
            if len(instantaneous_frequency) == len(y_analysis):
                plt.plot(times, instantaneous_frequency[::step] if len(y_analysis) > max_plot_points else instantaneous_frequency)
            else:
                # Create compatible data if lengths don't match
                plt.plot(times, np.zeros_like(y_plot))
            plt.title('Instantaneous Frequency')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            
            # Plot 3: Envelope Spectrum
            plt.subplot(3, 1, 3)
            plt.plot(env_freqs, env_fft)
            if am_mod_freq is not None:
                plt.axvline(am_mod_freq, color='r', linestyle='dashed', 
                        label=f'AM Mod Freq: {am_mod_freq:.2f} Hz')
            plt.title('Envelope Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            if am_mod_freq is not None:
                plt.legend()
        except Exception as e:
            print(f"WARNING: Error during modulation visualization: {e}")
            # Create a simple error message in the plot
            plt.clf()  # Clear the figure
            plt.text(0.5, 0.5, f"Visualization error: {str(e)}", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.results_dir, 'wow_signal_modulation.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return {
            "modulation_plot": output_path,
            "am_modulation_frequency": float(am_mod_freq) if am_mod_freq is not None else None,
            "am_modulation_strength": float(am_strength),
            "fm_modulation_depth": float(fm_mod_depth),
            "fm_modulation_strength": float(fm_strength),
            "predominant_modulation_type": mod_type,
            "modulation_index": float(fm_mod_depth / am_strength if am_strength > 0 else 0)
        }
    
    def generate_summary_report(self, results):
        """
        Generate a comprehensive summary report of the audio analysis
        
        Args:
            results: Dictionary with all analysis results
            
        Returns:
            Path to the saved summary report
        """
        output_path = os.path.join(self.results_dir, 'wow_signal_audio_analysis_summary.md')
        
        with open(output_path, 'w') as f:
            f.write("# Wow! Signal Audio Analysis Summary\n\n")
            
            f.write("## File Information\n\n")
            f.write(f"- **Filename:** {results['file_info']['filename']}\n")
            f.write(f"- **Duration:** {results['file_info']['duration']:.2f} seconds\n")
            f.write(f"- **Sample Rate:** {results['file_info']['sample_rate']} Hz\n")
            f.write(f"- **Number of Samples:** {results['file_info']['num_samples']}\n\n")
            
            # Include spectral analysis if available
            if 'spectral_analysis' in results and isinstance(results['spectral_analysis'], dict):
                if 'error' not in results['spectral_analysis']:
                    f.write("## Spectral Analysis\n\n")
                    f.write("### Key Spectral Features\n\n")
                    
                    if 'spectral_centroid' in results['spectral_analysis']:
                        f.write(f"- **Spectral Centroid:** {results['spectral_analysis']['spectral_centroid']:.2f} Hz\n")
                    
                    if 'spectral_bandwidth' in results['spectral_analysis']:
                        f.write(f"- **Spectral Bandwidth:** {results['spectral_analysis']['spectral_bandwidth']:.2f} Hz\n")
                    
                    if 'spectral_flatness' in results['spectral_analysis']:
                        f.write(f"- **Spectral Flatness:** {results['spectral_analysis']['spectral_flatness']:.4f} (0 = pure tone, 1 = white noise)\n")
                    
                    if 'spectral_rolloff' in results['spectral_analysis']:
                        f.write(f"- **Spectral Rolloff:** {results['spectral_analysis']['spectral_rolloff']:.2f} Hz\n\n")
                    
                    if 'top_peaks' in results['spectral_analysis']:
                        f.write("### Dominant Frequencies\n\n")
                        f.write("| Frequency (Hz) | Magnitude |\n")
                        f.write("|---------------|----------|\n")
                        for freq, mag in results['spectral_analysis']['top_peaks'][:5]:
                            f.write(f"| {freq:.2f} | {mag:.2f} |\n")
                        f.write("\n")
                else:
                    f.write("## Spectral Analysis\n\n")
                    f.write(f"Error during spectral analysis: {results['spectral_analysis'].get('error', 'Unknown error')}\n\n")
            
            f.write("### Visualizations\n\n")
            
            if results.get('waveform_path'):
                f.write(f"![Waveform]({os.path.basename(results['waveform_path'])})\n\n")
            
            if 'spectral_analysis' in results and 'spectrum_plot' in results['spectral_analysis']:
                f.write(f"![Spectrum]({os.path.basename(results['spectral_analysis']['spectrum_plot'])})\n\n")
            
            if results.get('spectrogram_path'):
                f.write(f"![Spectrogram]({os.path.basename(results['spectrogram_path'])})\n\n")
            
            if results.get('mel_spectrogram_path'):
                f.write(f"![Mel Spectrogram]({os.path.basename(results['mel_spectrogram_path'])})\n\n")
            
            if results.get('chromagram_path'):
                f.write(f"![Chromagram]({os.path.basename(results['chromagram_path'])})\n\n")
            
            # Pattern Analysis section
            if 'onset_analysis' in results and isinstance(results['onset_analysis'], dict) and 'error' not in results['onset_analysis']:
                f.write("## Pattern Analysis\n\n")
                
                f.write("### Onset Detection\n\n")
                f.write(f"- **Number of Detected Onsets:** {results['onset_analysis']['num_onsets']}\n")
                f.write(f"- **Onset Density:** {results['onset_analysis']['onset_density']:.4f} onsets/sec\n")
                if results['onset_analysis']['onset_regularity'] is not None:
                    f.write(f"- **Onset Regularity:** {results['onset_analysis']['onset_regularity']:.4f} seconds (std dev of inter-onset intervals)\n\n")
                else:
                    f.write("- **Onset Regularity:** Not enough onsets for analysis\n\n")
                    
                if 'onset_plot' in results['onset_analysis']:
                    f.write(f"![Onset Detection]({os.path.basename(results['onset_analysis']['onset_plot'])})\n\n")
            
            # Repetition and Periodicity section
            if 'pattern_analysis' in results and isinstance(results['pattern_analysis'], dict) and 'error' not in results['pattern_analysis']:
                f.write("### Repetition and Periodicity\n\n")
                
                if 'pattern_peaks' in results['pattern_analysis'] and len(results['pattern_analysis']['pattern_peaks']) > 0:
                    if 'estimated_periodicity' in results['pattern_analysis'] and results['pattern_analysis']['estimated_periodicity'] is not None:
                        f.write(f"- **Estimated Periodicity:** {results['pattern_analysis']['estimated_periodicity']:.3f} seconds\n")
                    else:
                        f.write("- **Estimated Periodicity:** Not detected\n")
                        
                    f.write("- **Detected Pattern Peaks at:** ")
                    for i, time in enumerate(results['pattern_analysis']['pattern_peaks'][:5]):
                        if i > 0:
                            f.write(", ")
                        f.write(f"{time:.3f}s")
                    f.write("\n")
                else:
                    f.write("- **No significant repeating patterns detected**\n")
                    
                if 'frequency_pattern_strength' in results['pattern_analysis']:
                    f.write(f"- **Frequency Pattern Strength:** {results['pattern_analysis']['frequency_pattern_strength']:.4f} (0 = random, 1 = perfect correlation)\n")
                
                if 'has_significant_patterns' in results['pattern_analysis']:
                    has_patterns = "Yes" if results['pattern_analysis']['has_significant_patterns'] else "No"
                    f.write(f"- **Contains Significant Patterns:** {has_patterns}\n\n")
                
                if 'autocorrelation_plot' in results['pattern_analysis']:
                    f.write(f"![Autocorrelation]({os.path.basename(results['pattern_analysis']['autocorrelation_plot'])})\n\n")
            
            if results.get('wavelet_path'):
                f.write(f"![Wavelet Transform]({os.path.basename(results['wavelet_path'])})\n\n")
            
            # Statistical Analysis section
            if 'statistical_analysis' in results and isinstance(results['statistical_analysis'], dict) and 'error' not in results['statistical_analysis']:
                stats = results['statistical_analysis']
                f.write("## Statistical Analysis\n\n")
                
                if 'mean_amplitude' in stats:
                    f.write(f"- **Mean Amplitude:** {stats['mean_amplitude']:.4f}\n")
                if 'std_dev' in stats:
                    f.write(f"- **Standard Deviation:** {stats['std_dev']:.4f}\n")
                if 'rms' in stats:
                    f.write(f"- **RMS Level:** {stats['rms']:.4f}\n")
                if 'peak_amplitude' in stats:
                    f.write(f"- **Peak Amplitude:** {stats['peak_amplitude']:.4f}\n")
                if 'dynamic_range_db' in stats:
                    f.write(f"- **Dynamic Range:** {stats['dynamic_range_db']:.2f} dB\n")
                if 'skewness' in stats:
                    f.write(f"- **Skewness:** {stats['skewness']:.4f} (0 = symmetrical distribution)\n")
                if 'kurtosis' in stats:
                    f.write(f"- **Kurtosis:** {stats['kurtosis']:.4f} (0 = normal distribution)\n")
                if 'mean_zero_crossing_rate' in stats:
                    f.write(f"- **Mean Zero-Crossing Rate:** {stats['mean_zero_crossing_rate']:.4f}\n")
                if 'spectral_entropy' in stats:
                    f.write(f"- **Spectral Entropy:** {stats['spectral_entropy']:.4f}\n")
                if 'temporal_variation' in stats:
                    f.write(f"- **Temporal Variation:** {stats['temporal_variation']:.4f}\n\n")
                
                if 'statistics_plot' in stats:
                    f.write(f"![Statistical Analysis]({os.path.basename(stats['statistics_plot'])})\n\n")
            
            # Modulation Analysis section
            if 'modulation_analysis' in results and isinstance(results['modulation_analysis'], dict) and 'error' not in results['modulation_analysis']:
                mod = results['modulation_analysis']
                f.write("## Modulation Analysis\n\n")
                
                # Handle possible missing keys
                if 'predominant_modulation_type' in mod:
                    f.write(f"- **Predominant Modulation Type:** {mod['predominant_modulation_type']}\n")
                    
                if 'am_modulation_strength' in mod:
                    f.write(f"- **AM Modulation Strength:** {mod['am_modulation_strength']:.4f}\n")
                
                if 'am_modulation_frequency' in mod:
                    if mod['am_modulation_frequency'] is not None:
                        f.write(f"- **AM Modulation Frequency:** {mod['am_modulation_frequency']:.2f} Hz\n")
                    else:
                        f.write(f"- **AM Modulation Frequency:** Not detected\n")
                    
                if 'fm_mod_depth' in mod:
                    f.write(f"- **FM Modulation Depth:** {mod['fm_mod_depth']:.4f}\n")
                    
                if 'fm_strength' in mod:
                    f.write(f"- **FM Modulation Strength:** {mod['fm_strength']:.4f}\n")
                    
                if 'modulation_index' in mod:
                    f.write(f"- **Modulation Index:** {mod['modulation_index']:.4f}\n\n")
                
                if 'modulation_plot' in mod:
                    f.write(f"![Modulation Analysis]({os.path.basename(mod['modulation_plot'])})\n\n")
            
            f.write("## Interpretation and Findings\n\n")
            
            # Determine signal characteristics based on analysis results
            if mod['predominant_modulation_type'] == "Predominantly AM":
                f.write("The Wow! Signal appears to predominantly use **amplitude modulation** to encode information. ")
            elif mod['predominant_modulation_type'] == "Predominantly FM":
                f.write("The Wow! Signal appears to predominantly use **frequency modulation** to encode information. ")
            else:
                f.write("The Wow! Signal shows characteristics of **mixed modulation** (both AM and FM components). ")
            
            if results['pattern_analysis']['has_significant_patterns']:
                f.write("There are **significant repeating patterns** in the signal that suggest ")
                if results['pattern_analysis'].get('estimated_periodicity'):
                    f.write(f"periodic behavior with approximately {results['pattern_analysis']['estimated_periodicity']:.3f} seconds between repetitions. ")
                else:
                    f.write("structured information rather than random noise. ")
            else:
                f.write("The signal shows limited or no clear repeating patterns, which could indicate ")
                f.write("either a non-periodic information encoding or a signal that doesn't contain structured information. ")
            
            if stats['spectral_entropy'] > 0.8:
                f.write("The high spectral entropy suggests a **complex and information-rich** signal ")
                f.write("rather than a simple carrier or tone. ")
            else:
                f.write("The relatively low spectral entropy suggests a **more tonal** signal ")
                f.write("with energy concentrated in specific frequency bands. ")
            
            # Conclude with assessment of likely origin
            f.write("\n\n### Possible Origin Assessment\n\n")
            
            # Make an assessment based on signal characteristics
            if (mod['predominant_modulation_type'] == "Predominantly AM" and
                stats['spectral_entropy'] < 0.7 and
                results['pattern_analysis']['has_significant_patterns']):
                
                f.write("Based on the analysis, the signal shows characteristics consistent with **human-designed communication systems**. ")
                f.write("The clear modulation pattern, structured periodicity, and spectral characteristics suggest ")
                f.write("a deliberately engineered signal rather than a natural phenomenon.")
                
            elif (mod['predominant_modulation_type'] == "Mixed" and
                 stats['spectral_entropy'] > 0.8 and
                 not results['pattern_analysis']['has_significant_patterns']):
                
                f.write("The signal's characteristics are more aligned with **natural phenomena or noise**. ")
                f.write("The high entropy, mixed modulation, and lack of clear patterns suggest ")
                f.write("this may be capturing cosmic background radiation, interference, or natural radio emissions.")
                
            else:
                f.write("The signal shows **mixed characteristics** that make definitive classification challenging. ")
                f.write("Some aspects suggest deliberate encoding (such as the modulation characteristics), ")
                f.write("while others point to potential natural origins or sophisticated signal processing. ")
                f.write("Additional context about the signal's acquisition would be needed for a more definitive conclusion.")
            
            f.write("\n\n---\n\n")
            f.write("*This analysis was performed using advanced signal processing techniques as part of ")
            f.write("the Wow! Signal Analysis Project.*\n")
            
        return output_path


def main():
    """Main entry point for the script"""
    print("\n" + "="*80)
    print(" "*30 + "WOW! SIGNAL AUDIO ANALYSIS")
    print("="*80 + "\n")
    
    # Path to the audio file
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'data', 'Wow_Signal_SETI_Project.mp3')
    
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found at {audio_path}")
            return 1
            
        # Create analyzer object with memory optimizations
        # Use lower sample rate and limit duration if needed
        analyzer = WowSignalAudioAnalysis(
            audio_path, 
            sr=22050,  # Use a moderate sample rate to reduce memory usage
            max_duration=300  # Limit to 5 minutes (adjust based on your file and system memory)
        )
        
        # Set the matplotlib backend to Agg (non-interactive) to prevent display issues
        import matplotlib
        matplotlib.use('Agg')
        
        # Lower dpi for all plots in the class
        plt.rcParams['figure.dpi'] = 150
        
        # Run the full analysis
        results = analyzer.run_full_analysis()
        
        print("\n" + "="*80)
        print(f"Analysis complete! Results saved to {analyzer.results_dir}")
        print("="*80 + "\n")
        
    except MemoryError:
        print("\nERROR: Out of memory while processing audio file.")
        print("Try reducing the file size or using a system with more RAM.")
        print("\n" + "="*80)
        print("Analysis failed due to memory limitations.")
        print("="*80 + "\n")
        return 1
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        print("Analysis failed.")
        print("="*80 + "\n")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
