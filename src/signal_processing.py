"""
Signal Processing Module for Wow! Signal Analysis

This script handles the basic signal processing and visualization of the Wow! signal data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import librosa.display
import pywt

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_wow_signal_data():
    """Load the Wow! signal data from the CSV file."""
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'data', 'wow_signal.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}. Run data_acquisition.py first.")
    
    return pd.read_csv(csv_path)

def interpolate_signal(time_or_df, intensity=None, target_points=1000, num_points=None):
    """
    Interpolate the Wow! signal to create a higher-resolution representation.
    This is needed for more detailed signal processing.
    
    Args:
        time_or_df: Either a DataFrame with the original signal data or a time array
        intensity: Intensity array (required if time_or_df is a time array)
        target_points: Number of points to interpolate to
        num_points: Alternative parameter name for target_points (for compatibility)
        
    Returns:
        Arrays with the interpolated time and intensity
    """
    # Handle different ways of calling this function
    if num_points is not None:
        target_points = num_points
        
    if isinstance(time_or_df, pd.DataFrame):
        df = time_or_df
        original_time = df['time'].values
        original_intensity = df['intensity'].values
    else:
        original_time = time_or_df
        if intensity is None:
            raise ValueError("If first argument is a time array, intensity must be provided")
        original_intensity = intensity
    
    # Create a finer time axis
    time_interp = np.linspace(np.min(original_time), np.max(original_time), target_points)
    
    # Interpolate intensity values
    intensity_interp = np.interp(time_interp, original_time, original_intensity)
    
    return time_interp, intensity_interp

def plot_signal_and_save(time, intensity, title, filename, annotations=None):
    """
    Create and save a plot of the signal.
    
    Args:
        time: Array of time points
        intensity: Array of signal intensity
        title: Plot title
        filename: Filename to save the plot to
        annotations: Optional dictionary of annotations {position: text}
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, intensity, linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Signal Intensity (SNR)", fontsize=12)
    plt.grid(True)
    
    # Add annotations if provided
    if annotations:
        for pos, text in annotations.items():
            idx = np.abs(time - pos).argmin()
            plt.annotate(text, (time[idx], intensity[idx]), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=12)
    
    # Save the figure
    project_root = get_project_root()
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def analyze_frequency_components(time, intensity, sample_rate=None):
    """
    Analyze the frequency components of the signal using FFT.
    
    Args:
        time: Array of time points
        intensity: Array of signal intensity
        sample_rate: Sample rate (if None, will be calculated from time)
        
    Returns:
        Tuple of (frequencies, amplitudes)
    """
    if sample_rate is None:
        # Calculate sample rate from time array
        sample_rate = len(time) / (time[-1] - time[0])
    
    # Perform FFT
    n = len(intensity)
    yf = fft(intensity)
    xf = fftfreq(n, 1/sample_rate)
    
    # Take the positive frequencies only
    xf = xf[:n//2]
    yf = 2.0/n * np.abs(yf[:n//2])
    
    return xf, yf

def create_spectrogram(time, intensity, sample_rate=None):
    """
    Create a spectrogram from the signal.
    
    Args:
        time: Array of time points
        intensity: Array of signal intensity
        sample_rate: Sample rate (if None, will be calculated from time)
        
    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    if sample_rate is None:
        # Calculate sample rate from time array
        sample_rate = len(time) / (time[-1] - time[0])
    
    # Calculate spectrogram
    frequencies, times, Sxx = signal.spectrogram(intensity, fs=sample_rate)
    
    return frequencies, times, Sxx

def perform_wavelet_transform(intensity):
    """
    Perform a wavelet transform on the signal.
    
    Args:
        intensity: Array of signal intensity
        
    Returns:
        Wavelet coefficients and associated information
    """
    # Compute continuous wavelet transform
    wavelet = 'morl'  # Morlet wavelet
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(intensity, scales, wavelet)
    
    return coeffs, freqs

def main():
    print("Starting Wow! signal processing and visualization...")
    
    # Load the data
    print("Loading data...")
    df = load_wow_signal_data()
    
    print("Data loaded successfully.")
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns}")
    print(df.head())
    
    # Original signal
    original_time = df['time'].values
    original_intensity = df['intensity'].values
    
    # Create interpolated signal for better visualization and analysis
    time_interp, intensity_interp = interpolate_signal(df, target_points=1000)
    
    # Plot interpolated signal
    annotations = {t: c for t, c in zip(original_time, ["6", "E", "Q", "U", "J", "5"])}
    plot_signal_and_save(
        time_interp, intensity_interp, 
        "Wow! Signal - Interpolated", 
        "wow_signal_interpolated.png",
        annotations
    )
    
    # Calculate sample rate from the interpolated data
    sample_rate = len(time_interp) / (time_interp[-1] - time_interp[0])
    
    # Frequency analysis
    frequencies, amplitudes = analyze_frequency_components(time_interp, intensity_interp, sample_rate)
    
    # Plot frequency components
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, amplitudes)
    plt.title("Frequency Components of Wow! Signal", fontsize=16)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True)
    plt.savefig('results/wow_signal_fft.png')
    plt.close()
    
    # Create spectrogram
    spec_freqs, spec_times, Sxx = create_spectrogram(time_interp, intensity_interp, sample_rate)
    
    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(spec_times, spec_freqs, 10 * np.log10(Sxx), shading='gouraud')
    plt.title("Spectrogram of Wow! Signal", fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.savefig('results/wow_signal_spectrogram.png')
    plt.close()
    
    # Wavelet analysis
    coeffs, freqs = perform_wavelet_transform(intensity_interp)
    
    # Plot wavelet transform
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coeffs), extent=[time_interp[0], time_interp[-1], freqs[-1], freqs[0]], 
               aspect='auto', cmap='viridis')
    plt.title("Wavelet Transform of Wow! Signal", fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Scale", fontsize=12)
    plt.colorbar(label='Magnitude')
    plt.savefig('results/wow_signal_wavelet.png')
    plt.close()
    
    print("Signal processing complete.")

if __name__ == "__main__":
    main()
