"""
Wow! Signal Analysis - Basic Script

This script performs a simplified analysis of the Wow! signal
detected on August 15, 1977.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import requests
from tqdm import tqdm
import time

# Define project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Wow! Signal Analysis")
print("===================")

# Step 1: Download the Wow! signal image
print("\nStep 1: Downloading Wow! signal image...")
url = 'https://upload.wikimedia.org/wikipedia/commons/d/d3/Wow_signal.jpg'
save_path = os.path.join(DATA_DIR, 'wow_signal.jpg')

try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    
    print(f"Downloaded image to {save_path}")
except Exception as e:
    print(f"Error downloading image: {e}")

# Step 2: Create Wow! signal data
print("\nStep 2: Creating Wow! signal dataset...")

# The character-to-intensity mapping
# Numbers 0-9 represent intensities 0-9 times the background level
# Letters A-Z represent intensities 10-35 times the background level
intensity_map = {
    **{str(i): i for i in range(10)},
    **{chr(i): i-55 for i in range(65, 91)}  # A=10, B=11, ..., Z=35
}

# The "6EQUJ5" sequence
wow_sequence = "6EQUJ5"

# Create time points (72 seconds total, divided into 6 observations)
# Each character corresponds to a 12-second interval
time_points = np.linspace(0, 72, len(wow_sequence))

# Convert to intensity values
intensity_values = [intensity_map[char] for char in wow_sequence]

# Create the DataFrame
df = pd.DataFrame({
    'time': time_points,
    'intensity': intensity_values,
    'character': list(wow_sequence),
    'channel': 2  # The signal was detected in channel 2
})

# Print the data
print("Wow! signal data:")
print(df)

# Save to CSV
csv_path = os.path.join(DATA_DIR, 'wow_signal.csv')
df.to_csv(csv_path, index=False)
print(f"Saved data to {csv_path}")

# Step 3: Visualize the original signal
print("\nStep 3: Creating visualizations...")
plt.figure(figsize=(12, 8))
plt.plot(df['time'], df['intensity'], 'o-', linewidth=2, markersize=10)
plt.title("Wow! Signal Intensity Over Time", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("Signal Intensity (SNR)", fontsize=14)
plt.grid(True)

# Add annotations for original characters
for i, row in df.iterrows():
    plt.annotate(f"{row['character']} ({int(row['intensity'])})", 
                (row['time'], row['intensity']), 
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=14)

plt.savefig(os.path.join(RESULTS_DIR, 'wow_signal_plot.png'))
print(f"Created plot: {os.path.join(RESULTS_DIR, 'wow_signal_plot.png')}")

# Step 4: Create an interpolated signal for analysis
print("\nStep 4: Creating interpolated signal...")
time_interp = np.linspace(df['time'].min(), df['time'].max(), 1000)
intensity_interp = np.interp(time_interp, df['time'], df['intensity'])

plt.figure(figsize=(12, 8))
plt.plot(df['time'], df['intensity'], 'o', markersize=10, label='Original Data Points')
plt.plot(time_interp, intensity_interp, '-', linewidth=2, label='Interpolated Signal')
plt.title("Wow! Signal - Original and Interpolated", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("Signal Intensity (SNR)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Add annotations
for i, row in df.iterrows():
    plt.annotate(row['character'], (row['time'], row['intensity']), 
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=14)

plt.savefig(os.path.join(RESULTS_DIR, 'wow_signal_interpolated.png'))
print(f"Created plot: {os.path.join(RESULTS_DIR, 'wow_signal_interpolated.png')}")

# Step 5: Frequency analysis
print("\nStep 5: Performing frequency analysis...")
sample_rate = len(time_interp) / (time_interp[-1] - time_interp[0])
n = len(intensity_interp)
yf = fft(intensity_interp)
xf = fftfreq(n, 1/sample_rate)

# Take the positive frequencies only
xf_pos = xf[:n//2]
yf_pos = 2.0/n * np.abs(yf[:n//2])

plt.figure(figsize=(12, 8))
plt.plot(xf_pos, yf_pos)
plt.title("Frequency Components of Wow! Signal", fontsize=16)
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'wow_signal_fft.png'))
print(f"Created plot: {os.path.join(RESULTS_DIR, 'wow_signal_fft.png')}")

# Step 6: Create a spectrogram
print("\nStep 6: Creating spectrogram...")
frequencies, times, Sxx = signal.spectrogram(intensity_interp, fs=sample_rate)

plt.figure(figsize=(12, 8))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.title("Spectrogram of Wow! Signal", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("Frequency (Hz)", fontsize=14)
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.savefig(os.path.join(RESULTS_DIR, 'wow_signal_spectrogram.png'))
print(f"Created plot: {os.path.join(RESULTS_DIR, 'wow_signal_spectrogram.png')}")

# Step 7: Pattern analysis
print("\nStep 7: Analyzing patterns in the signal...")
diffs = np.diff(intensity_values)
ratios = np.array([intensity_values[i+1]/intensity_values[i] for i in range(len(intensity_values)-1)])

print("\nSequential differences:", diffs)
print("Sequential ratios:", np.round(ratios, 2))

# Create a summary report
print("\nStep 8: Creating summary report...")
with open(os.path.join(RESULTS_DIR, 'summary.md'), 'w') as f:
    f.write("# Wow! Signal Analysis Summary\n\n")
    f.write("## Overview\n\n")
    f.write("This report summarizes the analysis of the Wow! signal detected on August 15, 1977.\n\n")
    
    f.write("## The Original Signal\n\n")
    f.write("The Wow! signal is represented by the sequence '6EQUJ5', where each character indicates the signal strength:\n\n")
    f.write("| Character | Signal Strength (× background) |\n")
    f.write("|-----------|------------------------------|\n")
    for char, intensity in zip(wow_sequence, intensity_values):
        f.write(f"| {char} | {intensity} |\n")
    f.write("\n")
    
    f.write("![Original Signal](wow_signal_plot.png)\n\n")
    
    f.write("## Signal Processing\n\n")
    f.write("We performed various signal processing techniques to analyze the characteristics of the signal.\n\n")
    
    f.write("### Interpolated Signal\n\n")
    f.write("![Interpolated Signal](wow_signal_interpolated.png)\n\n")
    
    f.write("### Frequency Analysis\n\n")
    f.write("![Frequency Components](wow_signal_fft.png)\n\n")
    
    f.write("### Time-Frequency Analysis\n\n")
    f.write("![Spectrogram](wow_signal_spectrogram.png)\n\n")
    
    f.write("## Pattern Analysis\n\n")
    f.write("We analyzed the sequence to look for potential patterns:\n\n")
    
    f.write("- Sequential differences: " + str(diffs.tolist()) + "\n")
    f.write("- Sequential ratios: " + str(np.round(ratios, 2).tolist()) + "\n\n")
    
    f.write("## Conclusions\n\n")
    f.write("The Wow! signal remains an intriguing astronomical mystery. Key observations:\n\n")
    
    f.write("1. The signal appeared at 1420.4556 MHz, near the hydrogen line frequency\n")
    f.write("2. The signal lasted for 72 seconds, which matches the transit time of a fixed point in space through the telescope's beam\n")
    f.write("3. The signal was narrowband (< 10 kHz), which is unusual for natural sources but consistent with technological signals\n")
    f.write("4. Despite repeated searches, the signal has never been detected again\n\n")
    
    f.write("Given the limited data available (essentially just 6 measurements), definitive conclusions about the signal's origin remain elusive.")

print("\nAnalysis complete! Results are available in the 'results' directory.")
print(f"Summary report: {os.path.join(RESULTS_DIR, 'summary.md')}")
