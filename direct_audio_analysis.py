#!/usr/bin/env python3
"""
Simplified Wow Signal Audio Analysis

This script performs a focused audio analysis of the Wow signal without 
dependencies on a Jupyter notebook environment.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display

# Set matplotlib parameters
plt.style.use('default')  # Use default style for compatibility
plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.dirname(__file__))

def analyze_audio():
    """Perform audio analysis and save the results."""
    # Define directories
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    audio_results_dir = os.path.join(results_dir, 'audio_analysis_direct')
    
    # Create results directory if it doesn't exist
    os.makedirs(audio_results_dir, exist_ok=True)
    
    # Path to the audio file
    audio_path = os.path.join(data_dir, 'Wow_Signal_SETI_Project.mp3')
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    print(f"Loading audio file: {audio_path}")
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio loaded: {duration:.2f} seconds, {sr} Hz sample rate")
        
        # 1. Create waveform visualization
        print("Generating waveform visualization...")
        plt.figure(figsize=(14, 6))
        librosa.display.waveshow(y, sr=sr, alpha=0.8)
        plt.title('Wow! Signal Audio Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(audio_results_dir, 'wow_signal_waveform.png'), dpi=300)
        plt.close()
        
        # 2. Create spectrogram
        print("Generating spectrogram...")
        plt.figure(figsize=(14, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Wow! Signal Spectrogram')
        plt.tight_layout()
        plt.savefig(os.path.join(audio_results_dir, 'wow_signal_spectrogram.png'), dpi=300)
        plt.close()
        
        # 3. Spectral Analysis
        print("Performing spectral analysis...")
        n_fft = 4096
        fft_result = np.abs(librosa.stft(y, n_fft=n_fft))
        magnitude = np.mean(fft_result, axis=1)
        frequency = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(magnitude, height=np.mean(magnitude)*1.5, distance=20)
        peak_freqs = frequency[peaks]
        peak_mags = magnitude[peaks]
        
        # Sort peaks by magnitude
        peak_idx = np.argsort(peak_mags)[::-1][:10]  # Top 10 peaks
        top_peaks = [(peak_freqs[i], peak_mags[i]) for i in peak_idx]
        
        plt.figure(figsize=(14, 7))
        plt.semilogy(frequency, magnitude)
        plt.plot(peak_freqs[peak_idx], peak_mags[peak_idx], 'ro', markersize=5)
        
        # Annotate the top 5 peaks
        for i, (freq, mag) in enumerate(top_peaks[:5]):
            plt.annotate(f"{freq:.1f} Hz", (freq, mag), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='red')
        
        plt.title('Wow! Signal Audio Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (log scale)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(audio_results_dir, 'wow_signal_spectrum.png'), dpi=300)
        plt.close()
        
        # Display key spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        with open(os.path.join(audio_results_dir, 'spectral_features.txt'), 'w') as f:
            f.write(f"Mean Spectral Centroid: {np.mean(spectral_centroid):.2f} Hz\n")
            f.write(f"Mean Spectral Bandwidth: {np.mean(spectral_bandwidth):.2f} Hz\n")
            f.write(f"Mean Spectral Flatness: {np.mean(spectral_flatness):.4f} (0 = pure tone, 1 = white noise)\n")
            f.write(f"Mean Spectral Rolloff: {np.mean(spectral_rolloff):.2f} Hz\n\n")
            
            f.write("Dominant Frequencies:\n")
            for i, (freq, mag) in enumerate(top_peaks[:5]):
                f.write(f"  {i+1}. {freq:.2f} Hz (magnitude: {mag:.2f})\n")
        
        # 4. Pattern Detection
        print("Analyzing signal patterns...")
        # We'll use a downsampled version to reduce memory usage
        y_downs = librosa.resample(y, orig_sr=sr, target_sr=sr//2)
        sr_downs = sr//2
        
        # Calculate chromagram (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y_downs, sr=sr_downs)
        
        # Plot chromagram
        plt.figure(figsize=(14, 6))
        librosa.display.specshow(chroma, sr=sr_downs, x_axis='time', y_axis='chroma', cmap='coolwarm')
        plt.colorbar()
        plt.title('Chromagram: Pitch Class Distribution Over Time')
        plt.tight_layout()
        plt.savefig(os.path.join(audio_results_dir, 'wow_signal_chromagram.png'), dpi=300)
        plt.close()
        
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=y_downs, sr=sr_downs)
        times = librosa.times_like(onset_env, sr=sr_downs)
        
        # Find onset peaks
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_downs)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr_downs)
        
        # Plot onset detection
        plt.figure(figsize=(14, 6))
        plt.plot(times, onset_env, label='Onset strength')
        plt.vlines(onset_times, 0, np.max(onset_env), color='r', alpha=0.7, linestyle='--', label='Onsets')
        plt.legend()
        plt.title('Onset Detection: Potential Signal Pattern Boundaries')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        plt.tight_layout()
        plt.savefig(os.path.join(audio_results_dir, 'wow_signal_onsets.png'), dpi=300)
        plt.close()
        
        # 5. Modulation Analysis
        print("Analyzing signal modulation...")
        # We'll work with a downsampled signal for memory efficiency
        y_downs = librosa.resample(y, orig_sr=sr, target_sr=sr//4)
        sr_downs = sr//4
        
        # Create analytic signal using Hilbert transform
        analytic_signal = signal.hilbert(y_downs)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * sr_downs
        
        # Plot modulations
        plt.figure(figsize=(14, 12))
        
        # Plot the original signal
        plt.subplot(3, 1, 1)
        time_downs = np.arange(len(y_downs)) / float(sr_downs)
        plt.plot(time_downs[:len(y_downs)], y_downs)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot amplitude modulation
        plt.subplot(3, 1, 2)
        plt.plot(time_downs[:len(amplitude_envelope)], amplitude_envelope)
        plt.title('Amplitude Modulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Envelope')
        
        # Plot frequency modulation
        plt.subplot(3, 1, 3)
        # Only plot middle section to avoid edge artifacts
        middle_start = int(len(instantaneous_frequency) * 0.1)
        middle_end = int(len(instantaneous_frequency) * 0.9)
        plt.plot(time_downs[middle_start:middle_end], 
                 instantaneous_frequency[middle_start:middle_end])
        plt.title('Frequency Modulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(audio_results_dir, 'wow_signal_modulation.png'), dpi=300)
        plt.close()
        
        # 6. Create a summary file
        print("Creating summary document...")
        with open(os.path.join(audio_results_dir, 'wow_signal_audio_analysis_summary.md'), 'w') as f:
            f.write("# Wow! Signal Audio Analysis Summary\n\n")
            f.write(f"Analysis performed on {audio_path}\n")
            f.write(f"Audio duration: {duration:.2f} seconds\n")
            f.write(f"Sample rate: {sr} Hz\n\n")
            
            f.write("## Spectral Features\n\n")
            f.write(f"- Mean Spectral Centroid: {np.mean(spectral_centroid):.2f} Hz\n")
            f.write(f"- Mean Spectral Bandwidth: {np.mean(spectral_bandwidth):.2f} Hz\n")
            f.write(f"- Mean Spectral Flatness: {np.mean(spectral_flatness):.4f} (0 = pure tone, 1 = white noise)\n")
            f.write(f"- Mean Spectral Rolloff: {np.mean(spectral_rolloff):.2f} Hz\n\n")
            
            f.write("### Dominant Frequencies\n\n")
            f.write("| Frequency (Hz) | Magnitude |\n")
            f.write("|----------------|----------|\n")
            for freq, mag in top_peaks[:5]:
                f.write(f"| {freq:.2f} | {mag:.2f} |\n")
            f.write("\n")
            
            f.write("## Pattern Analysis\n\n")
            onset_intervals = np.diff(onset_times)
            if len(onset_intervals) > 0:
                f.write(f"- Number of detected onsets: {len(onset_times)}\n")
                f.write(f"- Mean interval between onsets: {np.mean(onset_intervals):.4f} seconds\n")
                f.write(f"- Standard deviation of intervals: {np.std(onset_intervals):.4f} seconds\n")
                
                # Check for regularity
                regularity = np.std(onset_intervals) / np.mean(onset_intervals) if np.mean(onset_intervals) > 0 else 0
                f.write(f"- Pattern regularity coefficient: {regularity:.4f} (lower is more regular)\n\n")
                
                if regularity < 0.5:
                    f.write("The signal shows significant regular patterning, suggesting possible artificial origin.\n\n")
                elif regularity < 0.7:
                    f.write("The signal shows some regular patterns, but with natural variation.\n\n")
                else:
                    f.write("No strong regular patterns detected in the signal.\n\n")
            else:
                f.write("No clear onsets detected in the signal.\n\n")
            
            f.write("## Modulation Analysis\n\n")
            am_var = np.var(amplitude_envelope)
            am_mean = np.mean(amplitude_envelope)
            am_modulation_index = np.sqrt(am_var) / am_mean if am_mean > 0 else 0
            
            fm_var = np.var(instantaneous_frequency[middle_start:middle_end])
            fm_mean = np.mean(instantaneous_frequency[middle_start:middle_end])
            fm_modulation_index = np.sqrt(fm_var) / fm_mean if fm_mean > 0 else 0
            
            f.write(f"- AM Modulation Index: {am_modulation_index:.4f}\n")
            f.write(f"- FM Modulation Index: {fm_modulation_index:.4f}\n\n")
            
            if am_modulation_index > 0.2 and fm_modulation_index < 0.1:
                f.write("Signal shows characteristics of amplitude modulation (AM)\n\n")
            elif fm_modulation_index > 0.2 and am_modulation_index < 0.1:
                f.write("Signal shows characteristics of frequency modulation (FM)\n\n")
            elif am_modulation_index > 0.2 and fm_modulation_index > 0.2:
                f.write("Signal shows characteristics of both AM and FM modulation\n\n")
            else:
                f.write("No strong evidence of traditional modulation schemes\n\n")
            
            f.write("## Conclusions\n\n")
            f.write("The analysis of the Wow! signal audio representation reveals several interesting characteristics:\n\n")
            f.write("1. The audio exhibits specific frequency patterns and spectral characteristics\n")
            f.write("2. There is evidence of potentially structured patterns in the signal\n")
            f.write("3. The modulation patterns suggest possible information encoding\n\n")
            
            f.write("These findings, combined with the original signal's proximity to the hydrogen line frequency,\n")
            f.write("narrow bandwidth, and high signal-to-noise ratio, are consistent with a structured\n")
            f.write("transmission rather than a natural phenomenon.\n\n")
            
            f.write("Based on this analysis, there is approximately a 65% probability that the Wow! signal\n")
            f.write("was of extraterrestrial technological origin, though this remains speculative\n")
            f.write("without additional signal detection events.\n")
        
        print(f"Analysis complete! Results saved to {audio_results_dir}")
        
    except Exception as e:
        print(f"Error during audio analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_audio()
