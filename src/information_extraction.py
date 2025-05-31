"""
Information Extraction Module for Wow! Signal Analysis

This script attempts to identify potential patterns or encoded information in the Wow! signal.
It applies various decoding strategies and information theory techniques to explore
if there might be a message embedded in the signal.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import itertools

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

def interpolate_signal(df, target_points=1000):
    """
    Interpolate the Wow! signal to create a higher-resolution representation.
    
    Args:
        df: DataFrame with the original signal data
        target_points: Number of points to interpolate to
        
    Returns:
        Arrays with the interpolated time and intensity
    """
    original_time = df['time'].values
    original_intensity = df['intensity'].values
    
    # Create a finer time axis
    time_interp = np.linspace(original_time.min(), original_time.max(), target_points)
    
    # Interpolate intensity values
    intensity_interp = np.interp(time_interp, original_time, original_intensity)
    
    return time_interp, intensity_interp

def calculate_information_metrics(signal):
    """
    Calculate various information theory metrics for the signal.
    
    Args:
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of information metrics
    """
    results = {}
    
    # Discretize the signal for entropy calculation
    bins = min(20, len(signal) // 5)  # Rule of thumb for bin count
    hist, _ = np.histogram(signal, bins=bins)
    prob = hist / np.sum(hist)
    
    # Shannon Entropy
    entropy = -np.sum(prob * np.log2(prob + 1e-10))
    max_entropy = np.log2(bins)  # Maximum possible entropy for given bins
    results['shannon_entropy'] = entropy
    results['normalized_entropy'] = entropy / max_entropy
    
    # Calculate approximate complexity measures
    # Sample Entropy (simplified)
    diff_signal = np.diff(signal)
    mean_abs_diff = np.mean(np.abs(diff_signal))
    results['mean_abs_diff'] = mean_abs_diff
    
    # Lempel-Ziv complexity (simplified approximation)
    # Convert signal to binary sequence for complexity analysis
    median = np.median(signal)
    binary_signal = (signal > median).astype(int)
    
    # Count unique patterns of increasing length
    complexity = 0
    for length in range(1, min(8, len(binary_signal))):
        patterns = set()
        for i in range(len(binary_signal) - length + 1):
            pattern = tuple(binary_signal[i:i+length])
            patterns.add(pattern)
        complexity += len(patterns) / (2**length)
    
    results['lz_complexity_approx'] = complexity / (min(7, len(binary_signal) - 1))
    
    return results

def test_numerical_patterns(original_values):
    """
    Test for common numerical patterns in the signal values.
    
    Args:
        original_values: The original signal values (6EQUJ5 -> [6,14,26,30,19,5])
        
    Returns:
        Dictionary of pattern test results
    """
    results = {'detected_patterns': []}
    
    # Convert to numpy array if not already
    values = np.array(original_values)
    
    # Test for arithmetic sequence
    diffs = np.diff(values)
    if np.allclose(diffs, diffs[0], rtol=0.1, atol=2):
        results['detected_patterns'].append({
            'type': 'arithmetic_sequence',
            'common_difference': np.mean(diffs),
            'confidence': 'Medium'
        })
    
    # Test for geometric sequence
    ratios = values[1:] / (values[:-1] + 1e-10)  # Avoid division by zero
    if np.allclose(ratios, ratios[0], rtol=0.2, atol=0.5):
        results['detected_patterns'].append({
            'type': 'geometric_sequence',
            'common_ratio': np.mean(ratios),
            'confidence': 'Medium'
        })
    
    # Test for Fibonacci-like sequence
    fib_like = True
    for i in range(2, len(values)):
        expected = values[i-1] + values[i-2]
        if abs(values[i] - expected) / max(1, expected) > 0.3:  # Allow 30% tolerance
            fib_like = False
            break
    
    if fib_like and len(values) >= 3:
        results['detected_patterns'].append({
            'type': 'fibonacci_like',
            'confidence': 'Low'
        })
    
    # Test for prime number relationship
    is_prime = lambda n: n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))
    primes_count = sum(1 for v in values if is_prime(int(v)))
    
    if primes_count >= len(values) * 0.5:
        results['detected_patterns'].append({
            'type': 'prime_numbers',
            'prime_count': primes_count,
            'total_count': len(values),
            'confidence': 'Low'
        })
    
    # Test for common mathematical constants
    constants = {
        'pi': 3.14159,
        'e': 2.71828,
        'phi': 1.61803,
        'sqrt2': 1.41421,
        'sqrt3': 1.73205
    }
    
    for name, constant in constants.items():
        scaled_values = values * (constant / np.mean(values))
        error = np.mean(np.abs(scaled_values - np.round(scaled_values)))
        
        if error < 0.2:  # Arbitrary threshold
            results['detected_patterns'].append({
                'type': f'related_to_{name}',
                'scaling_factor': constant / np.mean(values),
                'error': error,
                'confidence': 'Very Low'
            })
    
    return results

def test_binary_encodings(values):
    """
    Test if the signal values could represent binary encoded information.
    
    Args:
        values: The signal intensity values
        
    Returns:
        Dictionary of binary encoding test results
    """
    results = {'possible_encodings': []}
    
    # Normalize values between 0 and 1
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    
    # Try binary thresholding at different levels
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        binary = (normalized > threshold).astype(int)
        binary_str = ''.join(map(str, binary))
        
        # Try to interpret as ASCII
        try:
            ascii_values = []
            for i in range(0, len(binary_str), 7):  # 7-bit ASCII
                if i + 7 <= len(binary_str):
                    byte = binary_str[i:i+7]
                    ascii_values.append(int(byte, 2))
            
            ascii_chars = [chr(v) if 32 <= v <= 126 else '?' for v in ascii_values]
            ascii_result = ''.join(ascii_chars)
            
            printable_ratio = sum(c != '?' for c in ascii_chars) / len(ascii_chars) if ascii_chars else 0
            
            if printable_ratio > 0.5:  # At least 50% printable characters
                results['possible_encodings'].append({
                    'type': 'ascii_7bit',
                    'threshold': threshold,
                    'binary': binary_str,
                    'result': ascii_result,
                    'printable_ratio': printable_ratio,
                    'confidence': 'Very Low'
                })
        except:
            pass
        
        # Try to interpret as ASCII (8-bit)
        try:
            ascii_values = []
            for i in range(0, len(binary_str), 8):  # 8-bit ASCII
                if i + 8 <= len(binary_str):
                    byte = binary_str[i:i+8]
                    ascii_values.append(int(byte, 2))
            
            ascii_chars = [chr(v) if 32 <= v <= 126 else '?' for v in ascii_values]
            ascii_result = ''.join(ascii_chars)
            
            printable_ratio = sum(c != '?' for c in ascii_chars) / len(ascii_chars) if ascii_chars else 0
            
            if printable_ratio > 0.5:  # At least 50% printable characters
                results['possible_encodings'].append({
                    'type': 'ascii_8bit',
                    'threshold': threshold,
                    'binary': binary_str,
                    'result': ascii_result,
                    'printable_ratio': printable_ratio,
                    'confidence': 'Very Low'
                })
        except:
            pass
    
    return results

def explore_amplitude_modulation(time, signal):
    """
    Explore if the amplitude modulation of the signal could encode information.
    
    Args:
        time: Array of time points
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of modulation analysis results
    """
    results = {}
    
    # Detect peaks and troughs
    # Simplified peak detection for demonstration
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(signal, height=np.mean(signal), distance=len(signal)/10)
    troughs, _ = find_peaks(-signal, height=-np.mean(signal), distance=len(signal)/10)
    
    results['num_peaks'] = len(peaks)
    results['num_troughs'] = len(troughs)
    
    # Calculate intervals between peaks
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks)
        results['peak_intervals'] = peak_intervals
        results['peak_intervals_std'] = np.std(peak_intervals)
        
        # Check if intervals are regular (potential indication of encoding)
        if np.std(peak_intervals) / np.mean(peak_intervals) < 0.2:  # Arbitrary threshold
            results['regular_intervals'] = True
            results['modulation_frequency'] = 1.0 / (np.mean(peak_intervals) * (time[1] - time[0]))
        else:
            results['regular_intervals'] = False
    
    # Check for potential on-off keying (OOK) pattern
    if len(peaks) + len(troughs) > 3:
        # Simplified approach: binarize the signal around its mean
        binary_signal = (signal > np.mean(signal)).astype(int)
        
        # Count transitions
        transitions = np.sum(np.abs(np.diff(binary_signal)))
        results['transitions'] = transitions
        
        # A high number of transitions relative to signal length might indicate encoding
        if transitions > len(signal) / 20:  # Arbitrary threshold
            results['potential_ook'] = True
        else:
            results['potential_ook'] = False
    
    return results

def analyze_signal_for_message(original_signal, interpolated_signal, original_time, interpolated_time):
    """
    Perform comprehensive analysis to detect any potential message or encoding.
    
    Args:
        original_signal: Original discrete signal values
        interpolated_signal: Interpolated continuous signal
        original_time: Original time points
        interpolated_time: Interpolated time points
        
    Returns:
        Dictionary of analysis results
    """
    results = {}
    
    # Calculate basic information metrics
    info_metrics = calculate_information_metrics(interpolated_signal)
    results['information_metrics'] = info_metrics
    
    # Test for common numerical patterns
    pattern_results = test_numerical_patterns(original_signal)
    results['numerical_patterns'] = pattern_results
    
    # Test for possible binary encodings
    binary_results = test_binary_encodings(interpolated_signal)
    results['binary_encodings'] = binary_results
    
    # Explore amplitude modulation
    modulation_results = explore_amplitude_modulation(interpolated_time, interpolated_signal)
    results['amplitude_modulation'] = modulation_results
    
    # Synthesize the findings
    summary = []
    
    if info_metrics['normalized_entropy'] < 0.7:
        summary.append(f"Signal has lower entropy than expected for random noise (normalized entropy: {info_metrics['normalized_entropy']:.2f}), suggesting potential structure")
    else:
        summary.append(f"Signal entropy (normalized: {info_metrics['normalized_entropy']:.2f}) is consistent with random or natural signals")
    
    if pattern_results['detected_patterns']:
        for pattern in pattern_results['detected_patterns']:
            summary.append(f"Detected potential {pattern['type']} pattern with {pattern['confidence']} confidence")
    else:
        summary.append("No clear numerical patterns detected in the signal values")
    
    if binary_results['possible_encodings']:
        summary.append(f"Found {len(binary_results['possible_encodings'])} potential binary encodings, but all with very low confidence")
    else:
        summary.append("No clear binary encoding patterns detected")
    
    if modulation_results.get('regular_intervals', False):
        summary.append(f"Detected regular modulation with frequency ~{modulation_results['modulation_frequency']:.4f} Hz")
    
    if modulation_results.get('potential_ook', False):
        summary.append("Signal exhibits characteristics that could be consistent with on-off keying (digital modulation)")
    
    results['summary'] = summary
    
    return results

def main():
    print("Starting Wow! signal information extraction analysis...")
    
    # Load the data
    df = load_wow_signal_data()
    
    # Get original values
    original_time = df['time'].values
    original_intensity = df['intensity'].values
    
    # Create interpolated signal for more detailed analysis
    time_interp, intensity_interp = interpolate_signal(df, target_points=1000)
    
    # Perform comprehensive analysis
    results = analyze_signal_for_message(original_intensity, intensity_interp, original_time, time_interp)
    
    # Save results to text file
    with open('results/information_extraction_results.txt', 'w') as f:
        f.write("Wow! Signal Information Extraction Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Information Metrics:\n")
        f.write("-" * 80 + "\n")
        for key, value in results['information_metrics'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Numerical Pattern Analysis:\n")
        f.write("-" * 80 + "\n")
        if results['numerical_patterns']['detected_patterns']:
            for pattern in results['numerical_patterns']['detected_patterns']:
                f.write(f"Pattern type: {pattern['type']}\n")
                for key, value in pattern.items():
                    if key != 'type':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        else:
            f.write("No significant numerical patterns detected.\n\n")
        
        f.write("Binary Encoding Analysis:\n")
        f.write("-" * 80 + "\n")
        if results['binary_encodings']['possible_encodings']:
            for encoding in results['binary_encodings']['possible_encodings']:
                f.write(f"Encoding type: {encoding['type']}\n")
                f.write(f"  Threshold: {encoding['threshold']}\n")
                f.write(f"  Binary: {encoding['binary']}\n")
                f.write(f"  Result: {encoding['result']}\n")
                f.write(f"  Printable ratio: {encoding['printable_ratio']}\n")
                f.write(f"  Confidence: {encoding['confidence']}\n\n")
        else:
            f.write("No likely binary encodings detected.\n\n")
        
        f.write("Amplitude Modulation Analysis:\n")
        f.write("-" * 80 + "\n")
        for key, value in results['amplitude_modulation'].items():
            if isinstance(value, np.ndarray):
                value = value.tolist()  # Convert numpy arrays to lists for printing
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Summary:\n")
        f.write("-" * 80 + "\n")
        for point in results['summary']:
            f.write(f"- {point}\n")
    
    # Create a visualization of the original "6EQUJ5" sequence with annotations
    plt.figure(figsize=(12, 8))
    
    # Plot the original signal points
    plt.plot(original_time, original_intensity, 'o-', markersize=10, label='Original Wow! Signal')
    
    # Add the original character annotations
    wow_chars = ["6", "E", "Q", "U", "J", "5"]
    for i, (t, intensity, char) in enumerate(zip(original_time, original_intensity, wow_chars)):
        plt.annotate(f"{char} ({int(intensity)})", 
                    (t, intensity),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center',
                    fontsize=12)
    
    # Add pattern annotations from the analysis
    if results['numerical_patterns']['detected_patterns']:
        pattern_text = []
        for pattern in results['numerical_patterns']['detected_patterns']:
            pattern_text.append(f"{pattern['type']} ({pattern['confidence']})")
        
        pattern_str = "\n".join(pattern_text)
        plt.figtext(0.15, 0.02, f"Detected patterns:\n{pattern_str}", 
                   bbox=dict(facecolor='yellow', alpha=0.2))
    
    plt.title("Wow! Signal Analysis: Potential Information Content", fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Signal Intensity (SNR)", fontsize=12)
    plt.grid(True)
    
    plt.savefig('results/information_extraction_analysis.png')
    plt.close()
    
    print("Information extraction analysis complete.")

if __name__ == "__main__":
    main()
