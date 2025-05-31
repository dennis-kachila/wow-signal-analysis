#!/usr/bin/env python3
"""
Main script for Wow! Signal Analysis

This script runs the complete analysis pipeline for the Wow! signal,
following the plan outlined in plan.md.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import the modules from our project
from src import data_acquisition
from src import signal_processing
from src import hypothesis_testing
from src import information_extraction

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.dirname(__file__))

def setup_directories():
    """Ensure required directories exist."""
    project_root = get_project_root()
    for dir_path in ['data', 'results', 'notebooks']:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)

def run_module(module_name, func_name='main'):
    """Run a specific function from a module."""
    print(f"\n{'=' * 80}")
    print(f"Running {module_name}.{func_name}()")
    print(f"{'=' * 80}\n")
    
    try:
        module = __import__(f'src.{module_name}', fromlist=[func_name])
        getattr(module, func_name)()
        print(f"\n{module_name} completed successfully!\n")
        return True
    except Exception as e:
        import traceback
        print(f"\nError in {module_name}: {e}\n")
        traceback.print_exc()
        return False

def create_summary_report():
    """Create a summary report of all analysis results."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open('results/summary_report.md', 'w') as f:
        f.write(f"# Wow! Signal Analysis Summary Report\n\n")
        f.write(f"Generated on: {timestamp}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the findings from a comprehensive re-analysis of the Wow! signal detected by the Big Ear radio telescope on August 15, 1977.\n\n")
        
        f.write("## Data Acquisition\n\n")
        f.write("The original Wow! signal is represented by the sequence '6EQUJ5', where each character represents the signal strength in a specific time bin:\n\n")
        f.write("- '6': 6 times the background level\n")
        f.write("- 'E': 14 times the background level\n")
        f.write("- 'Q': 26 times the background level\n")
        f.write("- 'U': 30 times the background level (the strongest point)\n")
        f.write("- 'J': 19 times the background level\n")
        f.write("- '5': 5 times the background level\n\n")
        
        f.write("![Wow Signal](results/wow_signal_initial_plot.png)\n\n")
        
        f.write("## Signal Processing\n\n")
        f.write("Various signal processing techniques were applied to analyze the signal's characteristics:\n\n")
        f.write("- Time-domain analysis\n")
        f.write("- Frequency-domain analysis using FFT\n")
        f.write("- Time-frequency analysis via spectrogram\n")
        f.write("- Multi-scale analysis using wavelets\n\n")
        
        f.write("![Interpolated Signal](results/wow_signal_interpolated.png)\n\n")
        f.write("![Signal Spectrogram](results/wow_signal_spectrogram.png)\n\n")
        f.write("![Wavelet Transform](results/wow_signal_wavelet.png)\n\n")
        
        f.write("## Hypothesis Testing\n\n")
        f.write("Three main hypotheses were evaluated for the origin of the signal:\n\n")
        
        # Read in the hypothesis testing results if available
        try:
            with open('results/hypothesis_testing_results.txt', 'r') as ht_file:
                hypothesis_results = ht_file.read()
                f.write("```\n")
                f.write(hypothesis_results)
                f.write("```\n\n")
        except:
            f.write("1. **Terrestrial Radio Frequency Interference (RFI)**: The signal could be from an Earth-based source\n")
            f.write("2. **Natural Astronomical Phenomenon**: The signal could be from a natural cosmic source\n")
            f.write("3. **Extraterrestrial Intelligence**: The signal could be artificial and of alien origin\n\n")
        
        f.write("![Hypothesis Comparison](results/hypothesis_comparison.png)\n\n")
        
        f.write("## Information Extraction\n\n")
        f.write("Analysis was performed to determine if the signal could contain encoded information:\n\n")
        
        # Read in the information extraction results if available
        try:
            with open('results/information_extraction_results.txt', 'r') as ie_file:
                lines = ie_file.readlines()
                summary_section = False
                summary_lines = []
                
                for line in lines:
                    if line.startswith("Summary:"):
                        summary_section = True
                        continue
                    if summary_section and line.startswith("-"):
                        continue
                    if summary_section and line.strip() and not line.startswith("="):
                        summary_lines.append(line)
                
                for line in summary_lines:
                    if line.strip():
                        f.write(f"{line}")
        except:
            f.write("- Statistical analysis of information content\n")
            f.write("- Testing for mathematical patterns\n")
            f.write("- Exploration of potential encoding schemes\n")
            f.write("- Analysis of amplitude modulation characteristics\n\n")
        
        f.write("![Information Analysis](results/information_extraction_analysis.png)\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("Based on our modern analysis of the Wow! signal:\n\n")
        
        f.write("1. The signal remains unusual and difficult to classify definitively.\n\n")
        
        f.write("2. The hydrogen line frequency (1420.4056 MHz) is significant. This frequency is protected from terrestrial use and has been theoretically proposed as an ideal frequency for interstellar communication.\n\n")
        
        f.write("3. The signal duration of 72 seconds exactly matches the time the telescope's beam would take to scan across a fixed point in space due to Earth's rotation, suggesting a celestial origin rather than a local source.\n\n")
        
        f.write("4. Despite extensive searches, the signal has never been detected again, which argues against both a persistent extraterrestrial beacon and most stable natural sources.\n\n")
        
        f.write("5. Modern information theory analysis reveals no conclusive evidence of encoded meaningful information, though the limited data (essentially just 6 data points) makes this determination inconclusive.\n\n")
        
        f.write("The signal remains one of the strongest candidates for a potential extraterrestrial transmission but still falls short of conclusive evidence. The true nature of the Wow! signal remains an open question in astronomy.\n\n")
        
        f.write("## Future Work\n\n")
        f.write("Additional research directions include:\n\n")
        f.write("1. Continued monitoring of the same sky region with more sensitive equipment\n")
        f.write("2. Application of advanced machine learning techniques to larger SETI datasets to identify similar signals\n")
        f.write("3. Development of more sophisticated encoding detection algorithms that can work with limited data points\n")
        f.write("4. Investigation of astronomical objects in the vicinity of the Wow! signal coordinates using modern telescopes\n")

    print(f"Summary report created: results/summary_report.md")
    
def main():
    """Run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(description='Wow! Signal Analysis')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip data download if already downloaded')
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    print("\n" + "="*80)
    print("Starting Wow! Signal Analysis Pipeline")
    print("="*80 + "\n")
    
    # Step 1: Data Acquisition
    print("\nPhase 1: Data Acquisition")
    print("-"*50)
    data_acquisition.main()
    
    # Step 2: Signal Processing
    print("\nPhase 2: Signal Processing")
    print("-"*50)
    signal_processing.main()
    
    # Step 3: Hypothesis Testing
    print("\nPhase 3: Hypothesis Testing")
    print("-"*50)
    hypothesis_testing.main()
    
    # Step 4: Information Extraction
    print("\nPhase 4: Information Extraction")
    print("-"*50)
    information_extraction.main()
    
    # Create summary report
    print("\nCreating Summary Report")
    print("-"*50)
    create_summary_report()
    
    print("\n" + "="*80)
    print("Wow! Signal Analysis Complete")
    print("Results are available in the 'results' directory")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
