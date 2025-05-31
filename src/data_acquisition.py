"""
Data Acquisition Module for Wow! Signal Analysis

This script handles the downloading and initial processing of the Wow! signal data.
Due to the historical nature of the data, we'll fetch the best available 
representations of the signal from online sources.
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from tqdm import tqdm
import io
import re

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def download_file(url, save_path):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL of the file to download
        save_path: Path where the file should be saved
    
    Returns:
        save_path if download successful, None otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download with progress bar
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
        
        print(f"Downloaded {os.path.basename(save_path)}")
        return save_path
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def download_wow_signal_data():
    """
    Download the best available data related to the Wow! signal.
    
    Note: Due to the historical nature of the signal and the lack of digital archives from 1977,
    we rely on later digitizations and reconstructions.
    
    Returns:
        Dictionary of downloaded file paths
    """
    project_root = get_project_root()
    base_data_dir = os.path.join(project_root, 'data')
    create_directory(base_data_dir)
    
    # Define URLs for data sources
    data_sources = {
        'original_printout': 'https://upload.wikimedia.org/wikipedia/commons/d/d3/Wow_signal.jpg',
        # Add more URLs as we find them
    }
    
    downloaded_files = {}
    
    for name, url in data_sources.items():
        file_ext = url.split('.')[-1]
        save_path = os.path.join(base_data_dir, f"{name}.{file_ext}")
        result = download_file(url, save_path)
        if result:
            downloaded_files[name] = result
    
    return downloaded_files

def create_wow_signal_dataframe():
    """
    Create a pandas DataFrame with the Wow! signal data.
    
    Since the original signal is represented as "6EQUJ5", we need to convert 
    these characters to actual signal-to-noise ratio values.
    
    Returns:
        DataFrame with time and intensity data
    """
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
        'channel': 2  # The signal was detected in channel 2
    })
    
    # Print the data
    print("Created Wow! signal dataframe:")
    print(df)
    
    # Save to CSV
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'data', 'wow_signal.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
    
    return df

def main():
    print("Starting Wow! signal data acquisition...")
    
    # Download available data files
    downloaded_files = download_wow_signal_data()
    
    # Create a structured DataFrame representation
    wow_df = create_wow_signal_dataframe()
    
    # Initial visualization
    plt.figure(figsize=(10, 6))
    plt.plot(wow_df['time'], wow_df['intensity'], 'o-', linewidth=2, markersize=8)
    plt.title("Wow! Signal Intensity Over Time", fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Signal Intensity (SNR)", fontsize=12)
    plt.grid(True)
    
    # Add annotations for original characters
    for i, char in enumerate(["6", "E", "Q", "U", "J", "5"]):
        plt.annotate(char, (wow_df['time'][i], wow_df['intensity'][i]), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12)
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/wow_signal_initial_plot.png')
    plt.close()
    
    print("Data acquisition complete.")
    
if __name__ == "__main__":
    main()
