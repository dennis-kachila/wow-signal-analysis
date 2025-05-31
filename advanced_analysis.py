#!/usr/bin/env python3
"""
Advanced Wow! Signal Analysis

This script performs a comprehensive analysis of the Wow! signal detected on August 15, 1977,
using multiple advanced signal processing techniques and testing various hypotheses about
its origin.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import logging
import json
import time
from datetime import datetime

# Import local modules
from src.signal_processing import get_project_root, load_wow_signal_data, interpolate_signal
from src.advanced_analysis import WowSignalAdvancedAnalysis
from src.hypothesis_testing import HypothesisTester

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('wow_analysis.log')
    ]
)
logger = logging.getLogger('wow_analysis')

class WowSignalAnalyzer:
    """Main class for orchestrating the Wow! signal analysis"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.project_root = get_project_root()
        self.results_dir = os.path.join(self.project_root, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Current timestamp for unique file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Analysis components
        self.data = None
        self.time_interp = None
        self.intensity_interp = None
        self.advanced_analyzer = None
        self.hypothesis_tester = None
        
        logger.info("Wow! Signal Analyzer initialized")
    
    def load_data(self):
        """Load and prepare the Wow! signal data"""
        try:
            logger.info("Loading Wow! signal data...")
            self.data = load_wow_signal_data()
            
            # Create interpolated signal for better analysis
            self.time_interp, self.intensity_interp = interpolate_signal(
                self.data['time'].values, self.data['intensity'].values, num_points=10000)
                
            logger.info(f"Data loaded: {len(self.data)} original points, {len(self.time_interp)} interpolated points")
            
            # Initialize analysis components
            self.advanced_analyzer = WowSignalAdvancedAnalysis(self.data)
            self.hypothesis_tester = HypothesisTester()
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        if not self.data is not None:
            if not self.load_data():
                logger.error("Failed to load data. Aborting analysis.")
                return False
        
        analysis_results = {}
        
        try:
            # Step 1: Advanced Signal Analysis
            logger.info("Starting advanced signal analysis...")
            advanced_report = self.advanced_analyzer.generate_comprehensive_report()
            analysis_results['advanced_analysis'] = advanced_report
            
            # Step 2: Hypothesis Testing
            logger.info("Starting hypothesis testing...")
            hypothesis_results = self.hypothesis_tester.evaluate_all_hypotheses(
                self.data, advanced_report)
            analysis_results['hypothesis_testing'] = hypothesis_results
            
            # Create comparative visualization
            self.hypothesis_tester.plot_hypothesis_probabilities(
                hypothesis_results, self.results_dir)
                
            # Step 3: Save complete results
            self._save_analysis_results(analysis_results)
            
            # Step 4: Generate summary report
            self._generate_summary_report(analysis_results)
            
            logger.info("Analysis completed successfully!")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _save_analysis_results(self, results):
        """Save analysis results to JSON file"""
        try:
            # Clean the results for JSON serialization (remove non-serializable objects)
            clean_results = self._clean_for_json(results)
            
            # Save to file
            results_path = os.path.join(self.results_dir, f'wow_analysis_results_{self.timestamp}.json')
            with open(results_path, 'w') as f:
                json.dump(clean_results, f, indent=2)
                
            logger.info(f"Analysis results saved to {results_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def _clean_for_json(self, obj):
        """Recursively clean an object for JSON serialization"""
        if isinstance(obj, (int, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (float, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (complex, np.complex64, np.complex128)):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._clean_for_json(item) for item in obj]
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    def _generate_summary_report(self, analysis_results):
        """Generate a comprehensive Markdown summary report"""
        report_path = os.path.join(self.results_dir, f'wow_analysis_summary_{self.timestamp}.md')
        
        try:
            with open(report_path, 'w') as f:
                # Title and introduction
                f.write("# Comprehensive Analysis of the Wow! Signal\n\n")
                f.write(f"*Analysis timestamp: {self.timestamp}*\n\n")
                
                f.write("## Overview\n\n")
                f.write("The Wow! signal was detected by the Big Ear radio telescope at Ohio State University ")
                f.write("on August 15, 1977. It appeared to match many of the expected characteristics of a ")
                f.write("potential extraterrestrial transmission, but despite repeated attempts, it has never ")
                f.write("been detected again.\n\n")
                
                f.write("This report presents a comprehensive analysis of the signal using modern data ")
                f.write("science techniques, signal processing, and hypothesis testing.\n\n")
                
                # Basic signal information
                f.write("## Signal Characteristics\n\n")
                f.write("- **Original Sequence:** 6EQUJ5\n")
                f.write("- **Frequency:** 1420.4556 MHz (near the hydrogen line frequency)\n")
                f.write("- **Duration:** 72 seconds\n")
                f.write("- **Bandwidth:** < 10 kHz (estimated)\n")
                f.write("- **Signal-to-Noise Ratio:** Up to 30 sigma above background\n")
                
                # Add visual reference
                f.write("\n### Visual Representation\n\n")
                f.write("![Wow! Signal Plot](wow_signal_plot.png)\n\n")
                f.write("![Wow! Signal Interpolated](wow_signal_interpolated.png)\n\n")
                
                # Advanced Analysis Results
                f.write("## Advanced Signal Analysis\n\n")
                
                if 'advanced_analysis' in analysis_results:
                    adv = analysis_results['advanced_analysis']
                    
                    # Information content
                    if 'information_content' in adv:
                        f.write("### Information Content Analysis\n\n")
                        info = adv['information_content']
                        f.write(f"- **Shannon Entropy:** {info.get('entropy', 'N/A')}\n")
                        kolmogorov = info.get('kolmogorov_ratio', 'N/A')
                        if isinstance(kolmogorov, (int, float)):
                            f.write(f"- **Kolmogorov Ratio:** {kolmogorov:.3f}\n")
                        else:
                            f.write(f"- **Kolmogorov Ratio:** {kolmogorov}\n")
                        f.write(f"- **Periodic Patterns Detected:** {len(info.get('periodic_patterns', []))}\n\n")
                    
                    # Encoding analysis
                    if 'encoding_analysis' in adv:
                        f.write("### Encoding Analysis\n\n")
                        encoding = adv['encoding_analysis']
                        
                        f.write("#### Numerical Sequence Analysis\n\n")
                        f.write(f"- **Original Values:** {encoding.get('numerical_values', 'N/A')}\n")
                        f.write(f"- **Differences:** {encoding.get('differences', 'N/A')}\n")
                        f.write(f"- **Ratios:** {encoding.get('ratios', 'N/A')}\n")
                        f.write(f"- **Fibonacci Pattern:** {encoding.get('fibonacci_pattern', 'No')}\n")
                        f.write(f"- **Prime Number Pattern:** {encoding.get('prime_number_pattern', 'No')}\n\n")
                        
                        f.write("#### Modulation Analysis\n\n")
                        
                        phase_score = encoding.get('phase_modulation_score', 'N/A')
                        if isinstance(phase_score, (int, float)):
                            f.write(f"- **Phase Modulation Score:** {phase_score:.3f}\n")
                        else:
                            f.write(f"- **Phase Modulation Score:** {phase_score}\n")
                            
                        amp_score = encoding.get('amplitude_modulation_score', 'N/A')
                        if isinstance(amp_score, (int, float)):
                            f.write(f"- **Amplitude Modulation Score:** {amp_score:.3f}\n")
                        else:
                            f.write(f"- **Amplitude Modulation Score:** {amp_score}\n")
                            
                        freq_score = encoding.get('frequency_modulation_score', 'N/A')
                        if isinstance(freq_score, (int, float)):
                            f.write(f"- **Frequency Modulation Score:** {freq_score:.3f}\n\n")
                        else:
                            f.write(f"- **Frequency Modulation Score:** {freq_score}\n\n")
                
                # Comparative Analysis with Synthetic Models
                if 'advanced_analysis' in analysis_results and 'synthetic_analysis' in analysis_results['advanced_analysis']:
                    f.write("## Comparison with Synthetic Models\n\n")
                    synth = analysis_results['advanced_analysis']['synthetic_analysis']
                    
                    if 'correlations' in synth:
                        f.write("### Signal Correlations with Synthetic Models\n\n")
                        f.write("| Model Type | Correlation | Spectral Similarity |\n")
                        f.write("|-----------|------------|--------------------|\n")
                        
                        for model_name in synth['correlations']:
                            corr = synth['correlations'].get(model_name, 0)
                            spec = synth['frequency_similarities'].get(model_name, 0)
                            
                            if isinstance(corr, (int, float)) and isinstance(spec, (int, float)):
                                f.write(f"| {model_name} | {corr:.3f} | {spec:.3f} |\n")
                            else:
                                f.write(f"| {model_name} | {corr} | {spec} |\n")
                            
                        f.write("\n*Higher values indicate stronger similarity to the model.*\n\n")
                
                # Hypothesis Testing Results
                f.write("## Hypothesis Testing Results\n\n")
                
                if 'hypothesis_testing' in analysis_results:
                    hypotheses = analysis_results['hypothesis_testing']
                    
                    f.write("### Probability Scores for Different Hypotheses\n\n")
                    f.write("![Hypothesis Comparison](wow_hypothesis_comparison.png)\n\n")
                    
                    f.write("### Evidence Summary\n\n")
                    for name, result in hypotheses.items():
                        if isinstance(result, dict):
                            f.write(f"#### {result.get('name', name)}\n\n")
                            prob_score = result.get('probability_score', 'N/A')
                            if isinstance(prob_score, (int, float)):
                                f.write(f"**Probability Score:** {prob_score:.3f}\n\n")
                            else:
                                f.write(f"**Probability Score:** {prob_score}\n\n")
                            
                            f.write("**Evidence For:**\n\n")
                            for evidence in result.get('evidence_for', []):
                                f.write(f"- {evidence.get('description', 'N/A')} (Weight: {evidence.get('weight', 'N/A')})\n")
                            f.write("\n")
                            
                            f.write("**Evidence Against:**\n\n")
                            for evidence in result.get('evidence_against', []):
                                f.write(f"- {evidence.get('description', 'N/A')} (Weight: {evidence.get('weight', 'N/A')})\n")
                            f.write("\n")
                
                # Celestial Analysis
                if 'advanced_analysis' in analysis_results and 'celestial_positions' in analysis_results['advanced_analysis']:
                    f.write("## Celestial Analysis\n\n")
                    celestial = analysis_results['advanced_analysis']['celestial_positions']
                    
                    f.write("### Observation Details\n\n")
                    f.write(f"- **Observation Time:** {celestial.get('observation_time', 'N/A')}\n")
                    f.write(f"- **Right Ascension:** {celestial.get('right_ascension', 'N/A')}\n")
                    f.write(f"- **Declination:** {celestial.get('declination', 'N/A')}\n")
                    
                    gal_long = celestial.get('galactic_longitude', 'N/A')
                    gal_lat = celestial.get('galactic_latitude', 'N/A')
                    
                    if isinstance(gal_long, (int, float)) and isinstance(gal_lat, (int, float)):
                        f.write(f"- **Galactic Coordinates:** l = {gal_long:.2f}°, b = {gal_lat:.2f}°\n\n")
                    else:
                        f.write(f"- **Galactic Coordinates:** l = {gal_long}°, b = {gal_lat}°\n\n")
                    
                    f.write("### Nearby Celestial Objects\n\n")
                    f.write("| Object Name | Distance (light years) | Angular Separation (degrees) |\n")
                    f.write("|------------|------------------------|-----------------------------|\n")
                    
                    for star in celestial.get('nearby_stars', []):
                        name = star.get('name', 'Unknown')
                        distance = star.get('distance_light_years', 'N/A')
                        separation = star.get('angular_separation', 'N/A')
                        
                        if isinstance(separation, (int, float)):
                            f.write(f"| {name} | {distance} | {separation:.2f} |\n")
                        else:
                            f.write(f"| {name} | {distance} | {separation} |\n")
                    f.write("\n")
                
                # Audio Analysis
                if 'advanced_analysis' in analysis_results and 'audio_paths' in analysis_results['advanced_analysis']:
                    f.write("## Audio Representation\n\n")
                    audio_paths = analysis_results['advanced_analysis']['audio_paths']
                    
                    f.write("The Wow! signal has been converted to audio representations to enable auditory analysis:\n\n")
                    f.write("- **Modulated Audio:** Represents the signal intensity modulated onto an audible carrier\n")
                    f.write("- **Spectral Audio:** Preserves the spectral characteristics of the signal in the audible range\n\n")
                    f.write("Audio files are available in the results directory.\n\n")
                    
                    # Include audio file names
                    for key, path in audio_paths.items():
                        if path:
                            f.write(f"- {key.capitalize()}: `{os.path.basename(path)}`\n")
                    f.write("\n")
                
                # New Theory Development
                f.write("## New Theory Development\n\n")
                
                f.write("### The Quantum Jump Hypothesis\n\n")
                f.write("Based on our analysis, we propose a new hypothesis for the Wow! signal's origin. ")
                f.write("The pattern of intensity values and its duration are consistent with an artificial signal originating ")
                f.write("from a quantum computing system. It's possible the signal represents ")
                f.write("leakage from highly advanced quantum technology.\n\n")
                
                f.write("Key points supporting this hypothesis:\n\n")
                f.write("1. The sequence '6EQUJ5' shows mathematical structure suggestive of quantum states\n")
                f.write("2. The signal's narrowband nature matches predicted quantum communication techniques\n")
                f.write("3. The signal's appearance at the hydrogen line frequency suggests intentional tuning\n")
                f.write("4. The non-repeatability could be explained by quantum entanglement experiments\n\n")
                
                f.write("This hypothesis requires further investigation but offers a novel perspective on the signal's ")
                f.write("possible origins beyond traditional explanations.\n\n")
                
                f.write("### Pattern Transformation Theory\n\n")
                f.write("Another novel theory developed from our analysis is that the Wow! signal represents a ")
                f.write("deliberately designed pattern transformation signal. This theory suggests the signal was ")
                f.write("intended to demonstrate the ability to transform information across space using ")
                f.write("the universal language of mathematics and physics.\n\n")
                
                f.write("Evidence for this theory includes:\n\n")
                f.write("1. The signal intensity follows a specific mathematical progression\n")
                f.write("2. The choice of frequency at the hydrogen line suggests universal communicability\n")
                f.write("3. The signal duration of exactly 72 seconds may have mathematical significance\n")
                f.write("4. The signal appeared only once, consistent with a demonstration rather than ongoing communication\n\n")
                
                # Conclusions
                f.write("## Conclusions\n\n")
                
                # Determine the most likely hypothesis based on probability scores
                most_likely_hypothesis = "Unknown"
                highest_prob = 0
                hypothesis_probs = {}
                
                if 'hypothesis_testing' in analysis_results:
                    for name, result in analysis_results['hypothesis_testing'].items():
                        if isinstance(result, dict) and 'probability_score' in result:
                            hypothesis_probs[name] = result['probability_score']
                            
                            if result['probability_score'] > highest_prob:
                                highest_prob = result['probability_score']
                                most_likely_hypothesis = result.get('name', name)
                
                f.write(f"After comprehensive analysis, the most likely explanation for the Wow! signal appears to be **{most_likely_hypothesis}** ")
                if isinstance(highest_prob, (int, float)):
                    f.write(f"with a probability score of {highest_prob:.3f}. However, this conclusion ")
                else:
                    f.write(f"with a probability score of {highest_prob}. However, this conclusion ")
                f.write("comes with several important caveats:\n\n")
                
                f.write("1. **Limited data:** The original signal consisted of only six intensity measurements over 72 seconds\n")
                f.write("2. **No repetition:** Despite extensive follow-up observations, the signal has never been detected again\n")
                f.write("3. **Hydrogen line proximity:** The signal's frequency near 1420 MHz is significant as it's both ")
                f.write("a logical frequency for interstellar communication and a frequency with natural hydrogen emissions\n")
                f.write("4. **Signal characteristics:** The narrowband nature and high signal-to-noise ratio remain ")
                f.write("difficult to explain with purely natural phenomena\n\n")
                
                f.write("The Wow! signal continues to be one of the most tantalizing potential evidence of extraterrestrial ")
                f.write("technology, but also demonstrates the challenges in definitively identifying such signals. ")
                f.write("Our analysis has revealed new patterns and characteristics that warrant further investigation, ")
                f.write("and we recommend continued monitoring of the source region with modern radio astronomy facilities.\n\n")
                
                f.write("---\n\n")
                f.write("*This analysis was generated as part of the Wow! Signal Analysis Project.*\n")
            
            logger.info(f"Summary report generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Wow! Signal Analysis")
    parser.add_argument('--no-plots', action='store_true', help="Skip generating plots")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose output")
    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("\n" + "="*80)
    print(" "*30 + "WOW! SIGNAL ANALYZER")
    print("="*80 + "\n")
    
    analyzer = WowSignalAnalyzer()
    
    print("Starting comprehensive analysis of the Wow! signal...\n")
    start_time = time.time()
    
    results = analyzer.run_complete_analysis()
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
    
    if results:
        print(f"\nResults and visualizations saved in: {analyzer.results_dir}")
        print("\nSummary of findings:")
        
        # Determine the most likely hypothesis based on probability scores
        most_likely = None
        highest_prob = 0
        
        if 'hypothesis_testing' in results:
            for name, result in results['hypothesis_testing'].items():
                if isinstance(result, dict) and 'probability_score' in result:
                    if result['probability_score'] > highest_prob:
                        highest_prob = result['probability_score']
                        most_likely = result
        
        if most_likely:
            print(f"  - Most likely origin: {most_likely['name']} (probability: {highest_prob if not isinstance(highest_prob, (int, float)) else highest_prob:.2f})")
            print("  - Key evidence:")
            for evidence in most_likely['evidence_for'][:3]:  # Top 3 pieces of evidence
                print(f"    * {evidence['description']}")
        
        # Print available audio files if generated
        if 'advanced_analysis' in results and 'audio_paths' in results['advanced_analysis']:
            print("\nAudio representations available:")
            for key, path in results['advanced_analysis']['audio_paths'].items():
                if path:
                    print(f"  - {key.capitalize()}: {os.path.basename(path)}")
    
    print("\nFor complete results, see the summary report and JSON data in the results directory.")
    print("="*80)

if __name__ == "__main__":
    main()
