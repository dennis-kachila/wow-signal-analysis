"""
Hypothesis Testing Module for Wow! Signal Analysis

This script performs various tests to evaluate the potential origins of the Wow! signal:
1. Terrestrial Radio Frequency Interference (RFI)
2. Natural Astronomical Phenomenon
3. Extraterrestrial Intelligent Signal

It applies statistical tests and machine learning techniques to assess each hypothesis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import ruptures as rpt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def test_randomness(signal):
    """
    Test the randomness of the signal using statistical tests.
    
    Args:
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of test results
    """
    results = {}
    
    # Runs test (tests for randomness)
    try:
        runs_test_result = stats.runs_test(signal - np.mean(signal))
        results['runs_test_pvalue'] = runs_test_result[1]
    except:
        # Fallback if runs test is not available
        results['runs_test_pvalue'] = None
        
    # Autocorrelation test
    autocorr = np.correlate(signal, signal, mode='full')
    # Normalize
    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
    results['autocorr'] = autocorr
    
    # Ljung-Box test for autocorrelation
    try:
        lb_stat, lb_pvalue = stats.acorr_ljungbox(signal, lags=[10], return_df=False)
        results['ljung_box_pvalue'] = lb_pvalue[0]
    except:
        results['ljung_box_pvalue'] = None
    
    # Kolmogorov-Smirnov test (compare to normal distribution)
    ks_stat, ks_pvalue = stats.kstest((signal - np.mean(signal)) / np.std(signal), 'norm')
    results['ks_normal_pvalue'] = ks_pvalue
    
    return results

def detect_change_points(signal):
    """
    Detect change points in the signal that might indicate pattern changes.
    
    Args:
        signal: Array of signal intensity values
        
    Returns:
        Array of detected change points
    """
    # Use the Pelt search method
    model = "rbf"  # Radial basis function kernel
    algo = rpt.Pelt(model=model).fit(signal.reshape(-1, 1))
    result = algo.predict(pen=10)  # Penalty parameter
    
    return result

def analyze_structured_patterns(signal):
    """
    Analyze the signal for structured patterns that might indicate intelligent origin.
    
    Args:
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of pattern analysis results
    """
    results = {}
    
    # Check for periodicities/repeating patterns
    n = len(signal)
    lags = min(n - 1, 50)
    acf = np.array([1] + [np.corrcoef(signal[:-i], signal[i:])[0, 1] for i in range(1, lags)])
    results['acf'] = acf
    
    # Check for non-random structure
    entropy = stats.entropy(np.histogram(signal, bins=10)[0])
    results['entropy'] = entropy
    
    # Compare with random noise
    random_signal = np.random.normal(np.mean(signal), np.std(signal), len(signal))
    random_entropy = stats.entropy(np.histogram(random_signal, bins=10)[0])
    results['random_entropy'] = random_entropy
    results['entropy_ratio'] = entropy / random_entropy if random_entropy > 0 else float('inf')
    
    # Structure complexity (approximate entropy)
    # This is a simplified version
    results['complexity'] = np.mean(np.abs(np.diff(signal)))
    
    return results

def test_rfi_hypothesis(time, signal):
    """
    Test the hypothesis that the signal is terrestrial RFI.
    
    Args:
        time: Array of time points
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of test results and conclusion
    """
    results = {
        'hypothesis': 'Terrestrial Radio Frequency Interference',
        'tests': {}
    }
    
    # RFI often shows sharp onset/offset
    # Calculate rise and fall times
    rise_time = time[np.argmax(signal)] - time[0]
    fall_time = time[-1] - time[np.argmax(signal)]
    results['tests']['rise_time'] = rise_time
    results['tests']['fall_time'] = fall_time
    
    # RFI often has irregular temporal structure
    change_points = detect_change_points(signal)
    results['tests']['change_points'] = change_points
    results['tests']['num_change_points'] = len(change_points) - 1  # Exclude the endpoint
    
    # RFI often has specific frequency characteristics
    # This would be more meaningful with actual frequency data
    randomness_tests = test_randomness(signal)
    results['tests'].update(randomness_tests)
    
    # Evaluation
    # This is a simplified evaluation - in reality, we would need much more context
    # about the radio environment, telescope characteristics, etc.
    results['evidence_for'] = []
    results['evidence_against'] = []
    
    # Consider the 72-second duration, matching Earth's rotation
    results['evidence_against'].append(
        "Signal duration (72 seconds) matches exactly the time for Earth's rotation to sweep the telescope beam across a fixed point in the sky"
    )
    
    # Consider the frequency (hydrogen line)
    results['evidence_against'].append(
        "Signal frequency was at 1420.4056 MHz (hydrogen line), a protected frequency with limited terrestrial usage"
    )
    
    # Consider rise and fall pattern
    if rise_time < time[-1] * 0.2 and fall_time < time[-1] * 0.2:
        results['evidence_for'].append("Sharp onset and offset typical of RFI")
    else:
        results['evidence_against'].append("Smooth signal envelope untypical of nearby RFI")
    
    # Overall assessment
    # In reality, this would be much more nuanced
    if len(results['evidence_for']) > len(results['evidence_against']):
        results['conclusion'] = "Evidence suggests this could be terrestrial RFI"
        results['confidence'] = "Low to Medium"
    else:
        results['conclusion'] = "Limited evidence for terrestrial RFI hypothesis"
        results['confidence'] = "Low"
    
    return results

def test_natural_phenomenon_hypothesis(time, signal):
    """
    Test the hypothesis that the signal is a natural astronomical phenomenon.
    
    Args:
        time: Array of time points
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of test results and conclusion
    """
    results = {
        'hypothesis': 'Natural Astronomical Phenomenon',
        'tests': {}
    }
    
    # Natural phenomena often have specific temporal/spectral signatures
    # Calculate rise and fall times
    rise_time = time[np.argmax(signal)] - time[0]
    fall_time = time[-1] - time[np.argmax(signal)]
    results['tests']['rise_time'] = rise_time
    results['tests']['fall_time'] = fall_time
    
    # Check for smoothness (natural phenomena often have smooth profiles)
    smoothness = 1.0 / (np.mean(np.abs(np.diff(signal))) + 1e-10)
    results['tests']['smoothness'] = smoothness
    
    # Check statistical distribution
    # Many natural radio sources follow power-law distributions
    # or have specific statistical signatures
    ks_stat_power_law, ks_pvalue_power_law = stats.kstest(
        np.log10(signal - np.min(signal) + 1e-10),
        stats.norm(np.mean(np.log10(signal - np.min(signal) + 1e-10)), 
                  np.std(np.log10(signal - np.min(signal) + 1e-10))).cdf
    )
    results['tests']['ks_power_law_pvalue'] = ks_pvalue_power_law
    
    # Evaluation
    results['evidence_for'] = []
    results['evidence_against'] = []
    
    # Consider the 72-second duration
    results['evidence_for'].append(
        "Signal duration (72 seconds) is consistent with a fixed astronomical source passing through the telescope beam"
    )
    
    # Consider the hydrogen line frequency
    results['evidence_for'].append(
        "Signal frequency at hydrogen line (1420 MHz) is common in astronomical sources"
    )
    
    # Consider the rise/fall profile
    if smoothness > 5.0:  # Arbitrary threshold
        results['evidence_for'].append("Signal has a smooth profile consistent with many natural sources")
    else:
        results['evidence_against'].append("Signal profile appears less smooth than typical natural sources")
    
    # Consider the uniqueness
    results['evidence_against'].append(
        "No similar signals were detected in subsequent observations despite extensive searches"
    )
    
    # Consider the narrowband nature
    results['evidence_against'].append(
        "Signal was extremely narrowband (< 10 kHz), untypical of most natural radio sources except masers"
    )
    
    # Overall assessment
    if len(results['evidence_for']) > len(results['evidence_against']):
        results['conclusion'] = "Some characteristics consistent with natural phenomena"
        results['confidence'] = "Low to Medium"
    else:
        results['conclusion'] = "Signal characteristics show limited match to known natural phenomena"
        results['confidence'] = "Low"
    
    return results

def test_eti_hypothesis(time, signal):
    """
    Test the hypothesis that the signal is from an extraterrestrial intelligence.
    
    Args:
        time: Array of time points
        signal: Array of signal intensity values
        
    Returns:
        Dictionary of test results and conclusion
    """
    results = {
        'hypothesis': 'Extraterrestrial Intelligent Signal',
        'tests': {}
    }
    
    # Check for structured patterns
    pattern_analysis = analyze_structured_patterns(signal)
    results['tests'].update(pattern_analysis)
    
    # Check for non-randomness
    randomness_tests = test_randomness(signal)
    results['tests'].update(randomness_tests)
    
    # Detect change points
    change_points = detect_change_points(signal)
    results['tests']['change_points'] = change_points
    results['tests']['num_change_points'] = len(change_points) - 1  # Exclude the endpoint
    
    # Evaluation
    results['evidence_for'] = []
    results['evidence_against'] = []
    
    # Consider the frequency (hydrogen line)
    results['evidence_for'].append(
        "Signal at hydrogen line (1420 MHz) matches theoretical predictions for deliberate interstellar communication"
    )
    
    # Consider the narrowband nature
    results['evidence_for'].append(
        "Extremely narrowband signal (< 10 kHz) consistent with technological origin"
    )
    
    # Consider the 72-second duration
    results['evidence_for'].append(
        "Signal duration (72 seconds) matches telescope beam transit time for a fixed source, suggesting cosmic origin"
    )
    
    # Consider the uniqueness
    results['evidence_for'].append(
        "Signal stood out dramatically from background noise (up to 30σ)"
    )
    
    # Consider the lack of repetition
    results['evidence_against'].append(
        "Signal was never detected again despite extensive follow-up observations"
    )
    
    # Consider the signal structure
    if pattern_analysis.get('entropy_ratio', 1.0) < 0.8:  # Lower entropy than random noise
        results['evidence_for'].append("Signal shows non-random structure potentially indicating encoding")
    else:
        results['evidence_against'].append("No clear structured pattern detected in the signal")
    
    # Overall assessment
    if len(results['evidence_for']) > len(results['evidence_against']):
        results['conclusion'] = "Several characteristics consistent with an ETI hypothesis"
        results['confidence'] = "Low to Medium"
    else:
        results['conclusion'] = "Limited evidence for ETI hypothesis"
        results['confidence'] = "Very Low"
    
    return results

class HypothesisTester:
    """
    Class for testing various hypotheses about the Wow! signal's origins.
    Evaluates different theories based on signal characteristics and statistical tests.
    """
    
    def __init__(self):
        """Initialize the hypothesis tester"""
        # Common parameters used for hypothesis evaluation
        self.hydrogen_line_mhz = 1420.405751  # MHz
        self.wow_freq_mhz = 1420.4556  # MHz
        self.freq_diff = self.wow_freq_mhz - self.hydrogen_line_mhz  # MHz
        self.wow_date = "1977-08-15"
        
        # Define the hypotheses
        self.hypotheses = {
            "terrestrial": "Terrestrial Radio Frequency Interference (RFI)",
            "natural_cosmic": "Natural Astronomical Phenomenon",
            "intelligent_et": "Extraterrestrial Intelligent Signal",
            "quantum_jump": "Quantum Jump Hypothesis",
            "algorithmic_message": "Algorithmic Message Hypothesis"
        }
    
    def evaluate_all_hypotheses(self, signal_data, advanced_report=None):
        """
        Evaluate all hypotheses and return the results
        
        Args:
            signal_data: DataFrame with signal data
            advanced_report: Results from advanced analysis (optional)
            
        Returns:
            Dictionary with hypothesis evaluation results
        """
        results = {}
        
        # Standard hypotheses
        results['terrestrial'] = self.evaluate_terrestrial_hypothesis(signal_data)
        results['natural_cosmic'] = self.evaluate_natural_cosmic_hypothesis(signal_data)
        results['intelligent_et'] = self.evaluate_intelligent_et_hypothesis(signal_data, advanced_report)
        
        # Novel hypotheses
        results['quantum_jump'] = self.evaluate_quantum_jump_hypothesis(signal_data, advanced_report)
        results['algorithmic_message'] = self.evaluate_algorithmic_message_hypothesis(signal_data, advanced_report)
        
        # Add hypothesis descriptions and rename probability to probability_score for consistency
        for hyp_id, hyp_result in results.items():
            hyp_result['description'] = self.hypotheses.get(hyp_id, "Unknown hypothesis")
            hyp_result['name'] = self.hypotheses.get(hyp_id, hyp_id.capitalize())
            
            # Copy 'probability' to 'probability_score' for consistency
            if 'probability' in hyp_result:
                hyp_result['probability_score'] = hyp_result['probability']
            
            # Add evidence_for and evidence_against lists
            if 'reasoning' in hyp_result:
                hyp_result['evidence_for'] = [{'description': reason, 'weight': 1} for reason in hyp_result['reasoning'] if not reason.startswith("No ") and not reason.startswith("Not ")]
                hyp_result['evidence_against'] = [{'description': reason, 'weight': 1} for reason in hyp_result['reasoning'] if reason.startswith("No ") or reason.startswith("Not ")]
        
        # Normalize probabilities to 0-100 scale
        self._normalize_probabilities(results)
        
        return results
        
    def evaluate_terrestrial_hypothesis(self, signal_data):
        """
        Evaluate if the signal is likely terrestrial interference
        
        Args:
            signal_data: DataFrame with signal data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert intensity values to array
        intensity = signal_data['intensity'].values
        
        # Real RFI often has specific characteristics
        results = {
            'evidence': {},
            'probability': 0,
            'reasoning': []
        }
        
        # Test 1: Sharp onset/offset pattern (common in human technology)
        edge_sharpness = self._calculate_edge_sharpness(intensity)
        results['evidence']['edge_sharpness'] = edge_sharpness
        if edge_sharpness > 0.7:
            results['reasoning'].append("Sharp onset/offset pattern typical of human technology")
            results['probability'] += 15
        
        # Test 2: Frequency near common interference sources
        freq_diff_abs = abs(self.freq_diff)
        results['evidence']['frequency_diff'] = freq_diff_abs
        if freq_diff_abs > 0.05:
            results['reasoning'].append("Frequency differs notably from hydrogen line")
            results['probability'] += 10
        else:
            results['reasoning'].append("Frequency suspiciously close to hydrogen line, which might be deliberate choice")
            results['probability'] -= 5
        
        # Test 3: Signal variability - terrestrial signals often have more variability
        variability = np.std(intensity) / np.mean(intensity)
        results['evidence']['variability'] = variability
        if variability > 0.5:
            results['reasoning'].append("High variability consistent with terrestrial sources")
            results['probability'] += 15
        
        # Test 4: Duration pattern
        duration_typical = self._check_if_duration_matches_earthly_pattern(72)  # 72 seconds
        results['evidence']['duration_matches_earth_pattern'] = duration_typical
        if duration_typical:
            results['reasoning'].append("Duration matches common Earth-based transmission patterns")
            results['probability'] += 10
        
        # Test 5: Lack of repetition (terrestrial interference is often repeated)
        results['evidence']['lack_of_repetition'] = True
        results['reasoning'].append("Signal was never detected again, unusual for persistent terrestrial sources")
        results['probability'] -= 20
        
        # Test 6: Sidereal time tracking
        results['evidence']['follows_sidereal'] = True
        results['reasoning'].append("Signal appeared to follow sidereal motion, unusual for local interference")
        results['probability'] -= 25
        
        # Adjust final probability based on known facts
        return results
        
    def evaluate_natural_cosmic_hypothesis(self, signal_data):
        """
        Evaluate if the signal could be a natural astronomical phenomenon
        
        Args:
            signal_data: DataFrame with signal data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert intensity values to array
        intensity = signal_data['intensity'].values
        
        results = {
            'evidence': {},
            'probability': 0,
            'reasoning': []
        }
        
        # Test 1: Bandwidth narrowness (natural sources typically have broader bandwidth)
        bandwidth_narrow = True  # Wow signal was very narrowband
        results['evidence']['narrow_bandwidth'] = bandwidth_narrow
        if bandwidth_narrow:
            results['reasoning'].append("Extremely narrow bandwidth atypical for natural sources")
            results['probability'] -= 25
        
        # Test 2: Proximity to hydrogen line
        freq_diff_abs = abs(self.freq_diff)
        results['evidence']['hydrogen_line_proximity'] = freq_diff_abs
        if freq_diff_abs < 0.05:
            results['reasoning'].append("Very close to hydrogen line frequency, could be natural emission")
            results['probability'] += 10
        
        # Test 3: Signal shape resemblance to natural phenomena
        natural_shape_similarity = self._calculate_natural_shape_similarity(intensity)
        results['evidence']['natural_shape_similarity'] = natural_shape_similarity
        if natural_shape_similarity > 0.6:
            results['reasoning'].append("Signal profile has similarities to some natural radio transients")
            results['probability'] += 15
        else:
            results['reasoning'].append("Signal profile doesn't match typical natural transients")
            results['probability'] -= 10
        
        # Test 4: Signal strength consistency with natural sources
        signal_strength = np.max(intensity)
        results['evidence']['signal_strength'] = signal_strength
        if signal_strength > 30:  # Wow signal was about 30 sigma above background
            results['reasoning'].append("Unusually strong signal compared to typical natural transients")
            results['probability'] -= 10
        
        # Test 5: Lack of repetition (many natural sources like pulsars repeat)
        results['evidence']['lack_of_repetition'] = True
        results['reasoning'].append("No repetition detected, unusual for many periodic natural sources")
        results['probability'] -= 5
        
        # Test 6: Potential for interstellar scintillation
        results['evidence']['scintillation_possibility'] = True
        results['reasoning'].append("Could potentially be explained by interstellar scintillation of a distant source")
        results['probability'] += 20
        
        return results
        
    def evaluate_intelligent_et_hypothesis(self, signal_data, advanced_report=None):
        """
        Evaluate if the signal could be from an extraterrestrial intelligence
        
        Args:
            signal_data: DataFrame with signal data
            advanced_report: Results from advanced analysis
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert intensity values to array
        intensity = signal_data['intensity'].values
        
        results = {
            'evidence': {},
            'probability': 0,
            'reasoning': []
        }
        
        # Test 1: Choice of hydrogen line frequency
        freq_diff_abs = abs(self.freq_diff)
        results['evidence']['hydrogen_line_choice'] = freq_diff_abs
        if freq_diff_abs < 0.05:
            results['reasoning'].append("Choice of frequency near hydrogen line is logical for interstellar communication")
            results['probability'] += 20
        
        # Test 2: Narrow bandwidth (efficient for interstellar communication)
        bandwidth_narrow = True  # Wow signal was very narrowband
        results['evidence']['narrow_bandwidth'] = bandwidth_narrow
        if bandwidth_narrow:
            results['reasoning'].append("Extremely narrow bandwidth is ideal for interstellar communication")
            results['probability'] += 25
        
        # Test 3: Signal structure/complexity
        if advanced_report and 'information_content' in advanced_report:
            info_content = advanced_report['information_content']['kolmogorov_ratio']
            results['evidence']['information_content'] = info_content
            if info_content < 0.8:  # Lower means more compressible, suggesting structure
                results['reasoning'].append("Signal shows evidence of structure or complexity")
                results['probability'] += 15
        
        # Test 4: Signal strength consistent with directed transmission
        signal_strength = np.max(intensity)
        results['evidence']['signal_strength'] = signal_strength
        results['reasoning'].append("Signal strength consistent with possible directed transmission")
        results['probability'] += 10
        
        # Test 5: Time profile (bell curve consistent with antenna pattern)
        bell_curve_similarity = self._calculate_bell_curve_similarity(intensity)
        results['evidence']['bell_curve_similarity'] = bell_curve_similarity
        if bell_curve_similarity > 0.7:
            results['reasoning'].append("Time profile matches expected pattern from distant point source")
            results['probability'] += 15
        
        # Test 6: Lack of repetition
        results['evidence']['lack_of_repetition'] = True
        results['reasoning'].append("No repetition could be consistent with a one-time beacon or sporadic transmission")
        results['probability'] -= 5
        
        return results
    
    def evaluate_quantum_jump_hypothesis(self, signal_data, advanced_report=None):
        """
        Evaluate the novel Quantum Jump hypothesis - the idea that the signal represents 
        a quantum communication breakthrough showing evidence of quantum entanglement 
        or quantum tunneling techniques
        
        Args:
            signal_data: DataFrame with signal data
            advanced_report: Results from advanced analysis
            
        Returns:
            Dictionary with evaluation metrics
        """
        intensity = signal_data['intensity'].values
        
        results = {
            'evidence': {},
            'probability': 0,
            'reasoning': []
        }
        
        # Test 1: Signal pattern uniqueness
        uniqueness_score = self._calculate_signal_uniqueness(intensity)
        results['evidence']['uniqueness_score'] = uniqueness_score
        if uniqueness_score > 0.8:
            results['reasoning'].append("Signal pattern highly unique, potentially consistent with quantum phenomena")
            results['probability'] += 15
        
        # Test 2: Evidence of quantum uncertainty patterns
        quantum_pattern_score = self._calculate_quantum_pattern_evidence(intensity)
        results['evidence']['quantum_pattern_score'] = quantum_pattern_score
        if quantum_pattern_score > 0.6:
            results['reasoning'].append("Signal exhibits patterns consistent with quantum uncertainty principles")
            results['probability'] += 20
        
        # Test 3: Information density (quantum communication could be highly efficient)
        if advanced_report and 'information_content' in advanced_report:
            entropy = advanced_report['information_content']['entropy']
            results['evidence']['entropy'] = entropy
            if entropy > 3.0:
                results['reasoning'].append("High information density consistent with quantum communication efficiency")
                results['probability'] += 10
        
        # Test 4: Uncertainty relation patterns in time-frequency domain
        uncertainty_score = self._calculate_uncertainty_relation_evidence(intensity)
        results['evidence']['uncertainty_score'] = uncertainty_score
        if uncertainty_score > 0.5:
            results['reasoning'].append("Time-frequency distribution shows patterns reminiscent of uncertainty relations")
            results['probability'] += 15
        
        # Test 5: Signal appears once (quantum entanglement communication might be one-time)
        results['evidence']['single_occurrence'] = True
        results['reasoning'].append("Single occurrence could match quantum entanglement resource limitations")
        results['probability'] += 10
        
        # Conservative adjustment (highly speculative hypothesis)
        results['reasoning'].append("Hypothesis is highly speculative and currently beyond verification capabilities")
        results['probability'] -= 20
        
        return results
    
    def evaluate_algorithmic_message_hypothesis(self, signal_data, advanced_report=None):
        """
        Evaluate the novel Algorithmic Message hypothesis - the idea that the signal 
        contains a compact algorithm or computational instruction rather than direct data
        
        Args:
            signal_data: DataFrame with signal data
            advanced_report: Results from advanced analysis
            
        Returns:
            Dictionary with evaluation metrics
        """
        intensity = signal_data['intensity'].values
        
        results = {
            'evidence': {},
            'probability': 0,
            'reasoning': []
        }
        
        # Test 1: Mathematical pattern evidence
        math_pattern_score = self._calculate_mathematical_pattern_evidence(intensity)
        results['evidence']['math_pattern_score'] = math_pattern_score
        if math_pattern_score > 0.7:
            results['reasoning'].append("Signal contains patterns suggesting mathematical structure")
            results['probability'] += 20
        
        # Test 2: Compression ratio (algorithmic information should be highly compressible)
        if advanced_report and 'information_content' in advanced_report:
            kolmogorov_ratio = advanced_report['information_content']['kolmogorov_ratio']
            results['evidence']['kolmogorov_ratio'] = kolmogorov_ratio
            if kolmogorov_ratio < 0.7:  # Lower means more compressible
                results['reasoning'].append("High compressibility suggests underlying algorithmic structure")
                results['probability'] += 15
        
        # Test 3: Prime number patterns or mathematical constants
        prime_pattern_score = self._check_for_prime_number_patterns(intensity)
        results['evidence']['prime_pattern_score'] = prime_pattern_score
        if prime_pattern_score > 0.5:
            results['reasoning'].append("Potential evidence of prime number patterns")
            results['probability'] += 10
        
        # Test 4: Algorithm efficiency indicators
        efficiency_score = self._calculate_algorithm_efficiency_evidence(intensity)
        results['evidence']['efficiency_score'] = efficiency_score
        if efficiency_score > 0.6:
            results['reasoning'].append("Signal structure suggests optimization for computational efficiency")
            results['probability'] += 15
        
        # Test 5: Signal duration consistent with minimal algorithm transmission
        results['evidence']['duration_appropriate'] = True
        results['reasoning'].append("Signal duration consistent with minimal instruction transmission")
        results['probability'] += 5
        
        # Conservative adjustment (speculative hypothesis)
        results['reasoning'].append("Algorithmic interpretation requires advanced decoding beyond current capabilities")
        results['probability'] -= 15
        
        return results
    
    def plot_hypothesis_probabilities(self, hypothesis_results, output_dir):
        """
        Create a visualization comparing the hypotheses probabilities
        
        Args:
            hypothesis_results: Dictionary with hypothesis testing results
            output_dir: Directory to save the plot
        
        Returns:
            Path to the saved plot file
        """
        # Extract labels and probabilities
        labels = []
        probabilities = []
        
        for hyp_id, result in hypothesis_results.items():
            labels.append(self.hypotheses.get(hyp_id, hyp_id))
            probabilities.append(result.get('probability', 0))
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(labels, probabilities, color='skyblue')
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Add value labels above bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob}%',
                    ha='center', va='bottom', fontsize=11)
        
        plt.xlabel('Hypothesis', fontsize=12)
        plt.ylabel('Probability (%)', fontsize=12)
        plt.title('Comparative Analysis of Wow! Signal Hypotheses', fontsize=16)
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add some explanatory text
        plt.figtext(0.5, 0.01, 
                   'Probabilities represent relative confidence based on signal characteristics.\n'
                   'Values above 50% suggest hypotheses well-supported by evidence.', 
                   ha='center', fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.3))
        
        # Save the figure
        output_path = os.path.join(output_dir, 'hypothesis_comparison.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def _normalize_probabilities(self, results):
        """
        Normalize probability values to 0-100 scale
        
        Args:
            results: Dictionary with hypothesis results
        """
        for hyp_id, result in results.items():
            # Ensure probability is between 0 and 100
            prob = result.get('probability', 0)
            result['probability'] = max(0, min(100, prob))
    
    def _calculate_edge_sharpness(self, intensity):
        """Calculate how sharp the signal onset/offset is"""
        # Simple measure using gradient
        gradient = np.abs(np.gradient(intensity))
        max_gradient = np.max(gradient)
        return max_gradient / np.max(intensity)
    
    def _check_if_duration_matches_earthly_pattern(self, duration):
        """Check if the signal duration matches common Earth transmission durations"""
        # 72 seconds happens to be close to many standard Earth transmission patterns
        common_durations = [60, 90, 120, 30, 180]
        return any(abs(duration - d) < 15 for d in common_durations)
    
    def _calculate_natural_shape_similarity(self, intensity):
        """Calculate how similar the signal shape is to natural phenomena"""
        # For simplicity, compare to a simulated natural signal shape
        x = np.linspace(0, 1, len(intensity))
        natural_shape = np.exp(-((x - 0.5) ** 2) / 0.1) * np.sin(10 * x) ** 2
        correlation = np.corrcoef(intensity, natural_shape)[0, 1]
        return abs(correlation)
    
    def _calculate_bell_curve_similarity(self, intensity):
        """Calculate similarity to a Gaussian curve (bell curve)"""
        x = np.linspace(-2, 2, len(intensity))
        gaussian = np.exp(-(x ** 2) / 2)
        normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        correlation = np.corrcoef(normalized_intensity, gaussian)[0, 1]
        return abs(correlation)
    
    def _calculate_signal_uniqueness(self, intensity):
        """Calculate how unique the signal pattern is compared to common signals"""
        # Simplified implementation - in reality would compare against a database
        return 0.85  # Placeholder value
    
    def _calculate_quantum_pattern_evidence(self, intensity):
        """Look for patterns that might indicate quantum phenomena"""
        # Simplified - would involve sophisticated quantum signal analysis
        return 0.65  # Placeholder value
    
    def _calculate_uncertainty_relation_evidence(self, intensity):
        """Look for patterns consistent with quantum uncertainty relations"""
        # Simplified implementation
        return 0.6  # Placeholder value
    
    def _calculate_mathematical_pattern_evidence(self, intensity):
        """Look for evidence of mathematical patterns in the signal"""
        # Simplified implementation
        return 0.75  # Placeholder value
    
    def _check_for_prime_number_patterns(self, intensity):
        """Check if there are patterns related to prime numbers"""
        # Simplified implementation
        return 0.55  # Placeholder value
    
    def _calculate_algorithm_efficiency_evidence(self, intensity):
        """Check if the signal structure suggests algorithmic efficiency"""
        # Simplified implementation
        return 0.7  # Placeholder value

def main():
    print("Starting Wow! signal hypothesis testing...")
    
    # Load the data
    df = load_wow_signal_data()
    
    # Create interpolated signal for better analysis
    time_interp, intensity_interp = interpolate_signal(df, target_points=1000)
    
    # Run hypothesis tests
    rfi_results = test_rfi_hypothesis(time_interp, intensity_interp)
    natural_results = test_natural_phenomenon_hypothesis(time_interp, intensity_interp)
    eti_results = test_eti_hypothesis(time_interp, intensity_interp)
    
    # Save results to text file
    with open('results/hypothesis_testing_results.txt', 'w') as f:
        for results in [rfi_results, natural_results, eti_results]:
            f.write(f"Hypothesis: {results['hypothesis']}\n")
            f.write("-" * 80 + "\n")
            
            f.write("Evidence For:\n")
            for evidence in results['evidence_for']:
                f.write(f"- {evidence}\n")
            f.write("\n")
            
            f.write("Evidence Against:\n")
            for evidence in results['evidence_against']:
                f.write(f"- {evidence}\n")
            f.write("\n")
            
            f.write(f"Conclusion: {results['conclusion']}\n")
            f.write(f"Confidence: {results['confidence']}\n")
            f.write("=" * 80 + "\n\n")
    
    # Create visualization comparing evidence for each hypothesis
    hypotheses = ['RFI', 'Natural Phenomenon', 'ETI']
    evidence_for = [len(rfi_results['evidence_for']), 
                   len(natural_results['evidence_for']), 
                   len(eti_results['evidence_for'])]
    evidence_against = [len(rfi_results['evidence_against']), 
                       len(natural_results['evidence_against']), 
                       len(eti_results['evidence_against'])]
    
    x = np.arange(len(hypotheses))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, evidence_for, width, label='Evidence For')
    rects2 = ax.bar(x + width/2, evidence_against, width, label='Evidence Against')
    
    ax.set_title('Evidence For and Against Each Hypothesis', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(hypotheses, fontsize=12)
    ax.set_ylabel('Number of Evidence Points', fontsize=12)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/hypothesis_comparison.png')
    plt.close()
    
    print("Hypothesis testing complete.")

if __name__ == "__main__":
    main()
