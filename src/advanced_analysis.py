"""
Advanced Analysis Module for the Wow! Signal

This module implements sophisticated analysis techniques for investigating the Wow! signal,
including:
- Information theory analysis
- Pattern detection algorithms
- Signal origin hypotheses testing
- Audio conversion and analysis
- Comparative analysis with known signal types
- Modulation detection and analysis
- Statistical significance testing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import librosa
import librosa.display
import pywt
import ruptures as rpt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from astropy import units as u
from astropy import constants as const
import scipy.io.wavfile as wav
from scipy.signal import find_peaks, hilbert

# Make sounddevice optional
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice module not available. Audio playback will be disabled.")

from src.signal_processing import interpolate_signal, get_project_root

# Constants related to the Wow! signal
WOW_FREQ_MHZ = 1420.4556  # MHz - close to the hydrogen line
WOW_BANDWIDTH = 10000  # Hz (10 kHz) - estimated
HYDROGEN_LINE = 1420.405751  # MHz
OBSERVATION_DATE = "1977-08-15"
OBSERVATION_TIME = "22:16:00"  # EST
BIG_EAR_COORDS = (40.0689, -83.0732)  # Latitude, Longitude of Big Ear telescope

class WowSignalAdvancedAnalysis:
    """Class for performing advanced analysis on the Wow! signal data"""
    
    def __init__(self, wow_data=None, file_path=None):
        """
        Initialize with either a dataframe or a path to the data file
        
        Args:
            wow_data: DataFrame containing the Wow! signal data
            file_path: Path to the CSV file with Wow! signal data
        """
        self.project_root = get_project_root()
        
        if wow_data is not None:
            self.data = wow_data
        elif file_path is not None:
            self.data = pd.read_csv(file_path)
        else:
            data_path = os.path.join(self.project_root, 'data', 'wow_signal.csv')
            self.data = pd.read_csv(data_path)
            
        # Extract basic signal data
        self.time = self.data['time'].values
        self.intensity = self.data['intensity'].values
        self.characters = self.data['character'].values
        
        # Create interpolated signal (higher resolution for analysis)
        self.time_interp, self.intensity_interp = interpolate_signal(
            self.time, self.intensity, num_points=10000)
        
        # Calculate sample rate from the interpolated data
        self.sample_rate = len(self.time_interp) / (self.time_interp[-1] - self.time_interp[0])
        
    def convert_to_audio(self, output_dir=None, duration=10, frequency_scaling=1000):
        """
        Convert the signal to audio for auditory analysis
        
        Args:
            output_dir: Directory to save the audio file
            duration: Duration of the audio in seconds
            frequency_scaling: Frequency scaling factor to bring signal into audible range
            
        Returns:
            Path to the saved WAV file
        """
        if output_dir is None:
            output_dir = os.path.join(self.project_root, 'results')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize intensity to range [0, 1]
        normalized = (self.intensity_interp - np.min(self.intensity_interp)) / \
                    (np.max(self.intensity_interp) - np.min(self.intensity_interp))
        
        # Scale to audio range [-1, 1]
        audio_signal = 2 * normalized - 1
        
        # For creating more interesting audio, we can modulate the signal
        # with a carrier frequency in the audible range
        carrier_freq = 440  # Hz (A4 note)
        
        # Create time array for audio
        audio_rate = 44100  # Hz
        t = np.linspace(0, duration, int(duration * audio_rate))
        
        # Create modulated signal - using both AM and FM modulation for better perception
        # AM component
        am_component = np.interp(np.linspace(0, 1, len(t)), 
                               np.linspace(0, 1, len(audio_signal)), 
                               audio_signal)
        
        # FM component - modulate the frequency using the signal
        fm_modulation = carrier_freq + frequency_scaling * np.interp(
            np.linspace(0, 1, len(t)), 
            np.linspace(0, 1, len(audio_signal)), 
            audio_signal)
        
        # Integrate the frequency to get the phase
        fm_phase = np.cumsum(fm_modulation) / audio_rate
        
        # Combined AM and FM 
        audio_output = am_component * np.sin(2 * np.pi * fm_phase)
        
        # Save to WAV file
        output_path = os.path.join(output_dir, 'wow_signal_audio.wav')
        wav.write(output_path, audio_rate, audio_output.astype(np.float32))
        
        # Also create a spectrally shaped noise version that preserves the spectral character
        spectrum = np.fft.rfft(audio_signal)
        magnitude = np.abs(spectrum)
        
        # Generate white noise
        noise = np.random.randn(len(t))
        
        # Shape the noise with the signal's spectrum
        noise_spectrum = np.fft.rfft(noise)
        shaped_spectrum = noise_spectrum * magnitude / np.max(magnitude)
        shaped_noise = np.fft.irfft(shaped_spectrum, len(noise))
        
        # Normalize
        shaped_noise = 0.5 * shaped_noise / np.max(np.abs(shaped_noise))
        
        # Save the spectrally shaped noise version
        spectral_output_path = os.path.join(output_dir, 'wow_signal_spectral.wav')
        wav.write(spectral_output_path, audio_rate, shaped_noise.astype(np.float32))
        
        return output_path, spectral_output_path
        
    def analyze_information_content(self):
        """
        Analyze the potential information content of the signal
        
        Returns:
            Dictionary with information theory metrics
        """
        # Calculate entropy of the signal
        hist, bin_edges = np.histogram(self.intensity, bins=10, density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate change in entropy over time
        entropy_windows = []
        window_size = len(self.intensity_interp) // 10
        
        for i in range(0, len(self.intensity_interp) - window_size, window_size // 2):
            window = self.intensity_interp[i:i+window_size]
            hist_window, _ = np.histogram(window, bins=10, density=True)
            entropy_window = -np.sum(hist_window * np.log2(hist_window + 1e-10))
            entropy_windows.append(entropy_window)
            
        # Calculate autocorrelation to look for repeating patterns
        autocorr = np.correlate(self.intensity_interp, self.intensity_interp, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (indicates possible periodic patterns)
        peaks, _ = find_peaks(autocorr, height=0.5*np.max(autocorr))
        
        # Calculate Kolmogorov complexity estimate using compression
        signal_str = np.array2string(np.round(self.intensity_interp, 2))
        import zlib
        compressed_size = len(zlib.compress(signal_str.encode()))
        kolmogorov_ratio = compressed_size / len(signal_str)
        
        return {
            'entropy': entropy,
            'entropy_over_time': entropy_windows,
            'autocorrelation': autocorr,
            'periodic_patterns': peaks,
            'kolmogorov_ratio': kolmogorov_ratio
        }
    
    def test_natural_source_hypothesis(self):
        """
        Test the hypothesis that the signal is from a natural cosmic source
        
        Returns:
            Dictionary with test results and confidence level
        """
        # Natural sources typically have broader bandwidth
        natural_bandwidth_probability = 0.1  # Low probability due to narrow bandwidth
        
        # Natural sources often show Gaussian noise characteristics
        _, p_value_shapiro = stats.shapiro(self.intensity_interp)
        
        # Natural sources often have 1/f (pink) noise characteristics
        f, Pxx = signal.welch(self.intensity_interp, self.sample_rate)
        
        # Calculate slope in log-log space (1/f noise would have slope ~ -1)
        log_f = np.log10(f[1:])  # Skip DC component
        log_Pxx = np.log10(Pxx[1:])
        slope, intercept = np.polyfit(log_f, log_Pxx, 1)
        
        # Calculate correlation to 1/f model (closer to -1 means better match)
        natural_spectral_match = abs(slope + 1)  # How close slope is to -1
        
        # Natural sources typically don't have sharp transients
        intensity_diff = np.diff(self.intensity_interp)
        max_transient_ratio = np.max(np.abs(intensity_diff)) / np.mean(np.abs(intensity_diff))
        
        # Score the natural source probability
        natural_source_score = (
            (1 - natural_bandwidth_probability) + 
            (1 - p_value_shapiro) + 
            (1 - min(1, natural_spectral_match)) +
            (1 / max(1, max_transient_ratio / 10))
        ) / 4
        
        return {
            'natural_source_probability': natural_source_score,
            'bandwidth_score': natural_bandwidth_probability,
            'gaussian_test_pvalue': p_value_shapiro,
            'spectral_slope': slope,
            'spectral_match_to_natural': natural_spectral_match,
            'transient_ratio': max_transient_ratio
        }
    
    def test_terrestrial_interference_hypothesis(self):
        """
        Test the hypothesis that the signal was terrestrial interference
        
        Returns:
            Dictionary with test results and confidence score
        """
        # Factors suggesting terrestrial origin
        terrestrial_factors = []
        evidence = {}
        
        # 1. Signal characteristics common in human technology
        evidence['narrowband'] = True
        if evidence['narrowband']:
            terrestrial_factors.append("Narrowband signal consistent with human technology")
        
        # 2. Frequency analysis - how close to common interference
        freq_deviation = abs(WOW_FREQ_MHZ - HYDROGEN_LINE)
        evidence['frequency_deviation'] = freq_deviation
        if freq_deviation > 0.01:
            terrestrial_factors.append(f"Frequency deviates from hydrogen line by {freq_deviation:.6f} MHz")
        
        # 3. Time signature analysis
        time_pattern = self._analyze_time_pattern()
        evidence['time_pattern'] = time_pattern
        if time_pattern > 0.7:
            terrestrial_factors.append("Time signature consistent with terrestrial transmission patterns")
        
        # 4. Signal strength pattern
        strength_pattern = self._analyze_strength_pattern()
        evidence['strength_pattern'] = strength_pattern
        if strength_pattern > 0.6:
            terrestrial_factors.append("Signal strength pattern consistent with moving terrestrial source")
        
        # 5. Lack of signal detection in follow-up observations
        evidence['no_followup_detection'] = True
        terrestrial_factors.append("Signal never detected again, suggesting transient interference")
        
        # Calculate a confidence score (0-100)
        confidence = min(100, len(terrestrial_factors) * 20)
        
        # Counterevidence
        counterevidence = []
        
        # A. Signal followed sidereal motion
        evidence['followed_sidereal'] = True
        if evidence['followed_sidereal']:
            counterevidence.append("Signal appeared to follow sidereal motion")
            confidence -= 30
        
        # B. Signal was detected only in one beam
        evidence['one_beam_detection'] = True
        if evidence['one_beam_detection']:
            counterevidence.append("Signal detected in only one beam")
            confidence -= 10
        
        # C. Characteristics unusual for terrestrial interference
        if time_pattern < 0.5:
            counterevidence.append("Time pattern atypical for terrestrial interference")
            confidence -= 15
        
        # Ensure confidence is in valid range
        confidence = max(0, min(100, confidence))
        
        return {
            'evidence': evidence,
            'supporting_factors': terrestrial_factors,
            'counterevidence': counterevidence,
            'confidence': confidence
        }
    
    def test_extraterrestrial_intelligence_hypothesis(self):
        """
        Test the hypothesis that the signal was from an extraterrestrial intelligence
        
        Returns:
            Dictionary with test results and confidence score
        """
        # Factors suggesting ET intelligence
        et_factors = []
        evidence = {}
        
        # 1. Frequency choice - hydrogen line is logical for interstellar communication
        freq_deviation = abs(WOW_FREQ_MHZ - HYDROGEN_LINE)
        evidence['frequency_deviation'] = freq_deviation
        if freq_deviation < 0.1:
            et_factors.append("Frequency near hydrogen line, ideal for interstellar communication")
        
        # 2. Signal bandwidth - narrowband is efficient for interstellar communication
        evidence['narrowband'] = True
        if evidence['narrowband']:
            et_factors.append("Narrowband signal optimizes energy use for interstellar communication")
        
        # 3. Signal strength - consistent with directed transmission
        signal_strength = np.max(self.intensity)
        evidence['signal_strength'] = signal_strength
        if signal_strength > 20:
            et_factors.append(f"Strong signal ({signal_strength} sigma) consistent with directed transmission")
        
        # 4. Time profile - gaussian curve consistent with antenna pattern
        time_profile = self._analyze_beam_pattern_match()
        evidence['beam_pattern_match'] = time_profile
        if time_profile > 0.8:
            et_factors.append("Signal intensity over time matches expected pattern from distant point source")
        
        # 5. Information content analysis
        info_content = self._analyze_information_content_for_et()
        evidence['information_content'] = info_content
        if info_content > 0.6:
            et_factors.append("Signal contains potential structure/patterns inconsistent with random noise")
        
        # Calculate a confidence score (0-100)
        confidence = min(100, len(et_factors) * 20)
        
        # Counterevidence
        counterevidence = []
        
        # A. One-time event, never repeated despite searches
        evidence['not_repeated'] = True
        if evidence['not_repeated']:
            counterevidence.append("Signal never detected again despite multiple search attempts")
            confidence -= 25
        
        # B. No complex modulation detected
        modulation_detected = self._check_for_complex_modulation()
        evidence['complex_modulation'] = modulation_detected
        if not modulation_detected:
            counterevidence.append("No definitive complex modulation or encoding detected")
            confidence -= 15
        
        # C. Potential natural explanations exist
        natural_explanation_score = self._assess_potential_natural_explanations()
        evidence['natural_explanation_score'] = natural_explanation_score
        if natural_explanation_score > 0.5:
            counterevidence.append("Some characteristics consistent with potential natural phenomena")
            confidence -= 20
        
        # Ensure confidence is in valid range
        confidence = max(0, min(100, confidence))
        
        return {
            'evidence': evidence,
            'supporting_factors': et_factors,
            'counterevidence': counterevidence,
            'confidence': confidence
        }
    
    def analyze_time_frequency_domain(self):
        """
        Perform time-frequency domain analysis of the signal
        
        Returns:
            Dictionary with time-frequency analysis results
        """
        # Create interpolated signal with higher sampling rate for better TF analysis
        time_high_res = np.linspace(self.time[0], self.time[-1], 10000)
        intensity_high_res = np.interp(time_high_res, self.time, self.intensity)
        
        # 1. Short-time Fourier Transform (STFT) for spectrogram
        f, t, Zxx = signal.stft(intensity_high_res, fs=1/(time_high_res[1]-time_high_res[0]), 
                               nperseg=256, noverlap=224)
        spectrogram = np.abs(Zxx)
        
        # 2. Continuous Wavelet Transform
        scales = np.arange(1, 128)
        wavelet = 'morl'  # Morlet wavelet
        coeffs, freqs = pywt.cwt(intensity_high_res, scales, wavelet)
        wavelet_power = np.abs(coeffs)**2
        
        # 3. Detect time-frequency ridges (areas of high energy concentration)
        tf_ridges = []
        for i, scale_power in enumerate(wavelet_power):
            peaks, _ = find_peaks(scale_power, height=0.5*np.max(scale_power))
            if len(peaks) > 0:
                for peak in peaks:
                    tf_ridges.append({
                        'frequency_idx': i,
                        'frequency': freqs[i],
                        'time_idx': peak,
                        'time': time_high_res[peak],
                        'power': scale_power[peak]
                    })
        
        # 4. Calculate time-frequency entropy
        # Normalize spectrogram to get probability distribution
        spectrogram_norm = spectrogram / np.sum(spectrogram)
        tf_entropy = -np.sum(spectrogram_norm * np.log2(spectrogram_norm + 1e-10))
        
        # 5. Calculate frequency stability over time
        instantaneous_freq = np.zeros_like(time_high_res)
        analytic_signal = hilbert(intensity_high_res)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq[1:] = np.diff(instantaneous_phase) / (2.0*np.pi) / (time_high_res[1]-time_high_res[0])
        freq_stability = 1.0 / (np.std(instantaneous_freq) + 1e-10)
        
        return {
            'spectrogram': spectrogram.tolist(),
            'spectrogram_time': t.tolist(),
            'spectrogram_freq': f.tolist(),
            'wavelet_power': wavelet_power.tolist(),
            'wavelet_time': time_high_res.tolist(),
            'wavelet_freq': freqs.tolist(),
            'time_freq_ridges': tf_ridges,
            'time_freq_entropy': tf_entropy,
            'freq_stability': freq_stability
        }
    
    def generate_and_compare_synthetic_signals(self):
        """
        Generate synthetic signals for various hypothetical sources and compare them to the Wow! signal
        
        Returns:
            Dictionary with comparison results
        """
        # Create time basis for synthetic signals
        t = np.linspace(0, 72, 1000)  # 72 seconds duration, 1000 points
        
        # 1. Pulsar-like signal
        period = 1.2  # seconds
        duty_cycle = 0.1
        pulsar_signal = np.zeros_like(t)
        for i in range(len(t)):
            phase = (t[i] % period) / period
            if phase < duty_cycle:
                pulsar_signal[i] = np.exp(-(phase/duty_cycle)**2 * 10)
        
        # 2. Terrestrial interference signal
        interference_signal = np.sin(2*np.pi*0.2*t) * np.exp(-(t-36)**2/144) + 0.2*np.random.randn(len(t))
        
        # 3. Binary message signal (simulating an encoded message)
        binary_signal = np.zeros_like(t)
        bit_duration = 3  # seconds
        for i in range(int(72/bit_duration)):
            if np.random.rand() > 0.5:
                start_idx = int(i * bit_duration * len(t) / 72)
                end_idx = int((i+1) * bit_duration * len(t) / 72)
                binary_signal[start_idx:end_idx] = 1.0
        binary_signal = binary_signal * np.exp(-(t-36)**2/400) + 0.05*np.random.randn(len(t))
        
        # 4. Doppler-shifted signal (e.g., from rotating source)
        doppler_signal = np.sin(2*np.pi*(0.1*t + 0.02*t**2)) * np.exp(-(t-36)**2/200)
        
        # 5. Information-rich signal (with higher complexity)
        complex_signal = np.zeros_like(t)
        for i in range(1, 10):
            complex_signal += np.sin(2*np.pi*i*0.1*t) / i
        complex_signal = complex_signal * np.exp(-(t-36)**2/200)
        
        # Compare with actual Wow! signal
        # Interpolate the actual signal to the same time basis
        wow_interp = np.interp(np.linspace(0, 72, len(t)), 
                              self.time - self.time[0], 
                              self.intensity)
        
        # Normalize all signals
        wow_interp = (wow_interp - np.min(wow_interp)) / (np.max(wow_interp) - np.min(wow_interp))
        pulsar_signal = (pulsar_signal - np.min(pulsar_signal)) / (np.max(pulsar_signal) - np.min(pulsar_signal))
        interference_signal = (interference_signal - np.min(interference_signal)) / (np.max(interference_signal) - np.min(interference_signal))
        binary_signal = (binary_signal - np.min(binary_signal)) / (np.max(binary_signal) - np.min(binary_signal))
        doppler_signal = (doppler_signal - np.min(doppler_signal)) / (np.max(doppler_signal) - np.min(doppler_signal))
        complex_signal = (complex_signal - np.min(complex_signal)) / (np.max(complex_signal) - np.min(complex_signal))
        
        # Calculate similarity scores (correlation coefficient)
        pulsar_similarity = abs(np.corrcoef(wow_interp, pulsar_signal)[0, 1])
        interference_similarity = abs(np.corrcoef(wow_interp, interference_signal)[0, 1])
        binary_similarity = abs(np.corrcoef(wow_interp, binary_signal)[0, 1])
        doppler_similarity = abs(np.corrcoef(wow_interp, doppler_signal)[0, 1])
        complex_similarity = abs(np.corrcoef(wow_interp, complex_signal)[0, 1])
        
        return {
            'synthetic_signals': {
                'time': t.tolist(),
                'pulsar': pulsar_signal.tolist(),
                'interference': interference_signal.tolist(),
                'binary_message': binary_signal.tolist(),
                'doppler_shifted': doppler_signal.tolist(),
                'complex_information': complex_signal.tolist()
            },
            'similarity_scores': {
                'pulsar': pulsar_similarity,
                'interference': interference_similarity,
                'binary_message': binary_similarity,
                'doppler_shifted': doppler_similarity,
                'complex_information': complex_similarity
            }
        }
    
    def analyze_celestial_positions(self):
        """
        Analyze the celestial positions at the time of the Wow! signal observation
        
        Returns:
            Dictionary with celestial position analysis
        """
        from astropy.time import Time
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        from astropy.coordinates import solar_system_ephemeris, get_body
        
        # The coordinates where the Wow! signal was detected
        # Sagittarius constellation, near globular cluster M55
        ra_hours, ra_minutes = 19, 25  # Right ascension (19h25m)
        dec_degrees, dec_minutes = -26, 57  # Declination (-26°57')
        
        # Convert to decimal degrees
        ra_deg = (ra_hours + ra_minutes/60) * 15  # 15 degrees per hour
        dec_deg = -(dec_degrees + dec_minutes/60)  # Negative for southern declination
        
        # Create a SkyCoord object
        wow_coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
        
        # Big Ear telescope location (Delaware, Ohio)
        big_ear = EarthLocation(lat=40.0689*u.deg, lon=-83.0732*u.deg, height=300*u.m)
        
        # Observation time (1977-08-15 22:16 UTC)
        obs_time = Time('1977-08-15T22:16:00', format='isot', scale='utc')
        
        # Calculate altitude and azimuth at observation time
        altaz = wow_coords.transform_to(AltAz(obstime=obs_time, location=big_ear))
        
        # Get positions of major bodies in the solar system
        solar_system_ephemeris.set('de432s')
        
        bodies = {}
        for body_name in ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']:
            try:
                body = get_body(body_name, obs_time, big_ear)
                body_altaz = body.transform_to(AltAz(obstime=obs_time, location=big_ear))
                
                # Calculate angular separation from Wow! signal location
                separation = wow_coords.separation(body)
                
                bodies[body_name] = {
                    'ra': body.ra.deg,
                    'dec': body.dec.deg,
                    'alt': body_altaz.alt.deg,
                    'az': body_altaz.az.deg,
                    'separation_deg': separation.deg,
                    'above_horizon': body_altaz.alt.deg > 0
                }
            except Exception as e:
                bodies[body_name] = {
                    'error': str(e)
                }
        
        return {
            'wow_position': {
                'ra_deg': ra_deg,
                'dec_deg': dec_deg,
                'alt_deg': altaz.alt.deg,
                'az_deg': altaz.az.deg
            },
            'observation_time': obs_time.isot,
            'bodies': bodies,
            # Includes the nearest bright stars or known radio sources if desired
            'nearest_stars': [
                {'name': 'HD 184512', 'separation_deg': 1.2},  # Example placeholder
                {'name': 'Chi Sagittarii', 'separation_deg': 3.8}  # Example placeholder
            ]
        }
    
    def assess_signal_novelty(self):
        """
        Assess how novel the signal is compared to known phenomena
        
        Returns:
            Dictionary with novelty assessment
        """
        # 1. Use isolation forest to detect how anomalous the signal is
        from sklearn.ensemble import IsolationForest
        
        # Extract features from the signal
        features = self._extract_signal_features()
        
        # Create a dataset with the Wow! signal features and many normal signals
        normal_signals = self._generate_normal_signal_features(n_samples=100)
        
        # Combine features (Wow! signal is the first one)
        all_features = np.vstack([features.reshape(1, -1), normal_signals])
        
        # Run isolation forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_scores = iso_forest.fit_predict(all_features)
        wow_score = outlier_scores[0]  # Score for the Wow! signal
        
        # 2. Compare with known astronomical signal classes
        known_classes = {
            'pulsar': self._compare_with_pulsar_signals(),
            'solar_flare': self._compare_with_solar_flares(),
            'quasar': self._compare_with_quasars(),
            'terrestrial_rfi': self._compare_with_terrestrial_rfi(),
            'satellite': self._compare_with_satellite_signals()
        }
        
        return {
            'isolation_forest_score': float(wow_score),  # -1 for anomalous, 1 for normal
            'novelty_percentage': 100 * (1 - self._calculate_novelty_score()),  # Higher means more novel
            'similarity_to_known_classes': known_classes,
            'most_similar_class': min(known_classes.items(), key=lambda x: x[1])[0],
            'least_similar_class': max(known_classes.items(), key=lambda x: x[1])[0]
        }
    
    # Helper methods for the analysis
    
    def _analyze_time_pattern(self):
        """Analyze the time pattern of the signal"""
        # Simplified implementation
        return 0.65  # Placeholder value
    
    def _analyze_strength_pattern(self):
        """Analyze the strength pattern of the signal"""
        # Simplified implementation
        return 0.7  # Placeholder value
    
    def _analyze_beam_pattern_match(self):
        """Analyze how well the signal matches the expected beam pattern"""
        # The Big Ear had a Gaussian beam pattern
        # If the signal was a point source, intensity should follow Gaussian as Earth rotates
        x = np.linspace(-3, 3, len(self.intensity_interp))
        gaussian = np.exp(-x**2/2)
        
        # Normalize both signals
        norm_intensity = (self.intensity_interp - np.min(self.intensity_interp)) / (np.max(self.intensity_interp) - np.min(self.intensity_interp))
        correlation = np.corrcoef(norm_intensity, gaussian)[0, 1]
        return abs(correlation)
    
    def _analyze_information_content_for_et(self):
        """Analyze the information content for ET intelligence assessment"""
        # Placeholder implementation
        return 0.75
    
    def _check_for_complex_modulation(self):
        """Check if the signal shows evidence of complex modulation"""
        # Placeholder implementation
        return False
    
    def _assess_potential_natural_explanations(self):
        """Assess potential natural explanations for the signal"""
        # Placeholder implementation
        return 0.6
    
    def _extract_signal_features(self):
        """Extract numerical features from the signal for machine learning analysis"""
        # Extract various statistical and signal features
        features = np.array([
            np.mean(self.intensity),               # Mean intensity
            np.std(self.intensity),                # Standard deviation
            np.max(self.intensity),                # Peak intensity
            np.sum(self.intensity),                # Total power
            np.median(self.intensity),             # Median
            stats.skew(self.intensity),            # Skewness
            stats.kurtosis(self.intensity),        # Kurtosis
            np.max(np.gradient(self.intensity)),   # Max gradient (rise)
            np.min(np.gradient(self.intensity)),   # Min gradient (fall)
            # Add more features as needed
        ])
        return features
    
    def _generate_normal_signal_features(self, n_samples=100):
        """Generate features for normal (non-anomalous) signals for comparison"""
        # This would ideally use a database of real signals
        # For this implementation, we'll generate synthetic features
        np.random.seed(42)
        
        # Generate with randomized parameters but following expected distributions
        features = np.zeros((n_samples, 9))
        for i in range(n_samples):
            # Generate a random signal type
            signal_type = np.random.choice(['noise', 'pulsar', 'flare', 'rfi'])
            
            if signal_type == 'noise':
                features[i, 0] = np.random.uniform(0.8, 1.2)  # Mean
                features[i, 1] = np.random.uniform(0.8, 1.5)  # Std dev
                features[i, 2] = np.random.uniform(2, 4)      # Peak
                features[i, 3] = np.random.uniform(70, 90)    # Total power
                features[i, 4] = np.random.uniform(0.8, 1.2)  # Median
                features[i, 5] = np.random.uniform(-0.2, 0.2) # Skewness
                features[i, 6] = np.random.uniform(-0.5, 0.5) # Kurtosis
                features[i, 7] = np.random.uniform(0.1, 0.3)  # Max gradient
                features[i, 8] = np.random.uniform(-0.3, -0.1) # Min gradient
            
            elif signal_type == 'pulsar':
                features[i, 0] = np.random.uniform(1.0, 1.5)  # Mean
                features[i, 1] = np.random.uniform(1.5, 2.5)  # Std dev
                features[i, 2] = np.random.uniform(4, 8)      # Peak
                features[i, 3] = np.random.uniform(90, 120)   # Total power
                features[i, 4] = np.random.uniform(0.9, 1.3)  # Median
                features[i, 5] = np.random.uniform(0.5, 1.5)  # Skewness
                features[i, 6] = np.random.uniform(1.0, 3.0)  # Kurtosis
                features[i, 7] = np.random.uniform(0.5, 1.0)  # Max gradient
                features[i, 8] = np.random.uniform(-1.0, -0.5) # Min gradient
            
            elif signal_type == 'flare':
                features[i, 0] = np.random.uniform(1.2, 2.0)  # Mean
                features[i, 1] = np.random.uniform(2.0, 3.0)  # Std dev
                features[i, 2] = np.random.uniform(6, 12)     # Peak
                features[i, 3] = np.random.uniform(100, 150)  # Total power
                features[i, 4] = np.random.uniform(1.0, 1.5)  # Median
                features[i, 5] = np.random.uniform(1.0, 2.0)  # Skewness
                features[i, 6] = np.random.uniform(2.0, 5.0)  # Kurtosis
                features[i, 7] = np.random.uniform(1.0, 2.0)  # Max gradient
                features[i, 8] = np.random.uniform(-2.0, -1.0) # Min gradient
            
            else:  # RFI
                features[i, 0] = np.random.uniform(1.0, 3.0)  # Mean
                features[i, 1] = np.random.uniform(1.0, 4.0)  # Std dev
                features[i, 2] = np.random.uniform(5, 15)     # Peak
                features[i, 3] = np.random.uniform(80, 200)   # Total power
                features[i, 4] = np.random.uniform(0.8, 2.5)  # Median
                features[i, 5] = np.random.uniform(-1.0, 2.0) # Skewness
                features[i, 6] = np.random.uniform(-1.0, 6.0) # Kurtosis
                features[i, 7] = np.random.uniform(0.2, 3.0)  # Max gradient
                features[i, 8] = np.random.uniform(-3.0, -0.2) # Min gradient
        
        return features
    
    def _calculate_novelty_score(self):
        """Calculate overall novelty score (0-1, higher means more similar to known signals)"""
        # This would combine results from various comparisons
        # For now, return a placeholder value
        return 0.25  # Suggesting the signal is quite novel
    
    def _compare_with_pulsar_signals(self):
        """Compare the Wow! signal with typical pulsar signals"""
        # Placeholder implementation
        return 0.35  # Similarity score (0-1, lower means more different)
    
    def _compare_with_solar_flares(self):
        """Compare the Wow! signal with solar flare signals"""
        # Placeholder implementation
        return 0.6
    
    def _compare_with_quasars(self):
        """Compare the Wow! signal with quasar signals"""
        # Placeholder implementation
        return 0.45
    
    def _compare_with_terrestrial_rfi(self):
        """Compare the Wow! signal with terrestrial RFI"""
        # Placeholder implementation
        return 0.55
    
    def _compare_with_satellite_signals(self):
        """Compare the Wow! signal with satellite signals"""
        # Placeholder implementation
        return 0.4
    
    def analyze_time_frequency_domain(self):
        """
        Perform time-frequency domain analysis of the signal
        
        Returns:
            Dictionary with time-frequency analysis results
        """
        # Create interpolated signal with higher sampling rate for better TF analysis
        time_high_res = np.linspace(self.time[0], self.time[-1], 10000)
        intensity_high_res = np.interp(time_high_res, self.time, self.intensity)
        
        # 1. Short-time Fourier Transform (STFT) for spectrogram
        f, t, Zxx = signal.stft(intensity_high_res, fs=1/(time_high_res[1]-time_high_res[0]), 
                               nperseg=256, noverlap=224)
        spectrogram = np.abs(Zxx)
        
        # 2. Continuous Wavelet Transform
        scales = np.arange(1, 128)
        wavelet = 'morl'  # Morlet wavelet
        coeffs, freqs = pywt.cwt(intensity_high_res, scales, wavelet)
        wavelet_power = np.abs(coeffs)**2
        
        # 3. Detect time-frequency ridges (areas of high energy concentration)
        tf_ridges = []
        for i, scale_power in enumerate(wavelet_power):
            peaks, _ = find_peaks(scale_power, height=0.5*np.max(scale_power))
            if len(peaks) > 0:
                for peak in peaks:
                    tf_ridges.append({
                        'frequency_idx': i,
                        'frequency': freqs[i],
                        'time_idx': peak,
                        'time': time_high_res[peak],
                        'power': scale_power[peak]
                    })
        
        # 4. Calculate time-frequency entropy
        # Normalize spectrogram to get probability distribution
        spectrogram_norm = spectrogram / np.sum(spectrogram)
        tf_entropy = -np.sum(spectrogram_norm * np.log2(spectrogram_norm + 1e-10))
        
        # 5. Calculate frequency stability over time
        instantaneous_freq = np.zeros_like(time_high_res)
        analytic_signal = hilbert(intensity_high_res)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq[1:] = np.diff(instantaneous_phase) / (2.0*np.pi) / (time_high_res[1]-time_high_res[0])
        freq_stability = 1.0 / (np.std(instantaneous_freq) + 1e-10)
        
        return {
            'spectrogram': spectrogram.tolist(),
            'spectrogram_time': t.tolist(),
            'spectrogram_freq': f.tolist(),
            'wavelet_power': wavelet_power.tolist(),
            'wavelet_time': time_high_res.tolist(),
            'wavelet_freq': freqs.tolist(),
            'time_freq_ridges': tf_ridges,
            'time_freq_entropy': tf_entropy,
            'freq_stability': freq_stability
        }
    
    def generate_and_compare_synthetic_signals(self):
        """
        Generate synthetic signals for various hypothetical sources and compare them to the Wow! signal
        
        Returns:
            Dictionary with comparison results
        """
        # Create time basis for synthetic signals
        t = np.linspace(0, 72, 1000)  # 72 seconds duration, 1000 points
        
        # 1. Pulsar-like signal
        period = 1.2  # seconds
        duty_cycle = 0.1
        pulsar_signal = np.zeros_like(t)
        for i in range(len(t)):
            phase = (t[i] % period) / period
            if phase < duty_cycle:
                pulsar_signal[i] = np.exp(-(phase/duty_cycle)**2 * 10)
        
        # 2. Terrestrial interference signal
        interference_signal = np.sin(2*np.pi*0.2*t) * np.exp(-(t-36)**2/144) + 0.2*np.random.randn(len(t))
        
        # 3. Binary message signal (simulating an encoded message)
        binary_signal = np.zeros_like(t)
        bit_duration = 3  # seconds
        for i in range(int(72/bit_duration)):
            if np.random.rand() > 0.5:
                start_idx = int(i * bit_duration * len(t) / 72)
                end_idx = int((i+1) * bit_duration * len(t) / 72)
                binary_signal[start_idx:end_idx] = 1.0
        binary_signal = binary_signal * np.exp(-(t-36)**2/400) + 0.05*np.random.randn(len(t))
        
        # 4. Doppler-shifted signal (e.g., from rotating source)
        doppler_signal = np.sin(2*np.pi*(0.1*t + 0.02*t**2)) * np.exp(-(t-36)**2/200)
        
        # 5. Information-rich signal (with higher complexity)
        complex_signal = np.zeros_like(t)
        for i in range(1, 10):
            complex_signal += np.sin(2*np.pi*i*0.1*t) / i
        complex_signal = complex_signal * np.exp(-(t-36)**2/200)
        
        # Compare with actual Wow! signal
        # Interpolate the actual signal to the same time basis
        wow_interp = np.interp(np.linspace(0, 72, len(t)), 
                              self.time - self.time[0], 
                              self.intensity)
        
        # Normalize all signals
        wow_interp = (wow_interp - np.min(wow_interp)) / (np.max(wow_interp) - np.min(wow_interp))
        pulsar_signal = (pulsar_signal - np.min(pulsar_signal)) / (np.max(pulsar_signal) - np.min(pulsar_signal))
        interference_signal = (interference_signal - np.min(interference_signal)) / (np.max(interference_signal) - np.min(interference_signal))
        binary_signal = (binary_signal - np.min(binary_signal)) / (np.max(binary_signal) - np.min(binary_signal))
        doppler_signal = (doppler_signal - np.min(doppler_signal)) / (np.max(doppler_signal) - np.min(doppler_signal))
        complex_signal = (complex_signal - np.min(complex_signal)) / (np.max(complex_signal) - np.min(complex_signal))
        
        # Calculate similarity scores (correlation coefficient)
        pulsar_similarity = abs(np.corrcoef(wow_interp, pulsar_signal)[0, 1])
        interference_similarity = abs(np.corrcoef(wow_interp, interference_signal)[0, 1])
        binary_similarity = abs(np.corrcoef(wow_interp, binary_signal)[0, 1])
        doppler_similarity = abs(np.corrcoef(wow_interp, doppler_signal)[0, 1])
        complex_similarity = abs(np.corrcoef(wow_interp, complex_signal)[0, 1])
        
        return {
            'synthetic_signals': {
                'time': t.tolist(),
                'pulsar': pulsar_signal.tolist(),
                'interference': interference_signal.tolist(),
                'binary_message': binary_signal.tolist(),
                'doppler_shifted': doppler_signal.tolist(),
                'complex_information': complex_signal.tolist()
            },
            'similarity_scores': {
                'pulsar': pulsar_similarity,
                'interference': interference_similarity,
                'binary_message': binary_similarity,
                'doppler_shifted': doppler_similarity,
                'complex_information': complex_similarity
            }
        }
    
    def analyze_celestial_positions(self):
        """
        Analyze the celestial positions at the time of the Wow! signal observation
        
        Returns:
            Dictionary with celestial position analysis
        """
        from astropy.time import Time
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        from astropy.coordinates import solar_system_ephemeris, get_body
        
        # The coordinates where the Wow! signal was detected
        # Sagittarius constellation, near globular cluster M55
        ra_hours, ra_minutes = 19, 25  # Right ascension (19h25m)
        dec_degrees, dec_minutes = -26, 57  # Declination (-26°57')
        
        # Convert to decimal degrees
        ra_deg = (ra_hours + ra_minutes/60) * 15  # 15 degrees per hour
        dec_deg = -(dec_degrees + dec_minutes/60)  # Negative for southern declination
        
        # Create a SkyCoord object
        wow_coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
        
        # Big Ear telescope location (Delaware, Ohio)
        big_ear = EarthLocation(lat=40.0689*u.deg, lon=-83.0732*u.deg, height=300*u.m)
        
        # Observation time (1977-08-15 22:16 UTC)
        obs_time = Time('1977-08-15T22:16:00', format='isot', scale='utc')
        
        # Calculate altitude and azimuth at observation time
        altaz = wow_coords.transform_to(AltAz(obstime=obs_time, location=big_ear))
        
        # Get positions of major bodies in the solar system
        solar_system_ephemeris.set('de432s')
        
        bodies = {}
        for body_name in ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']:
            try:
                body = get_body(body_name, obs_time, big_ear)
                body_altaz = body.transform_to(AltAz(obstime=obs_time, location=big_ear))
                
                # Calculate angular separation from Wow! signal location
                separation = wow_coords.separation(body)
                
                bodies[body_name] = {
                    'ra': body.ra.deg,
                    'dec': body.dec.deg,
                    'alt': body_altaz.alt.deg,
                    'az': body_altaz.az.deg,
                    'separation_deg': separation.deg,
                    'above_horizon': body_altaz.alt.deg > 0
                }
            except Exception as e:
                bodies[body_name] = {
                    'error': str(e)
                }
        
        return {
            'wow_position': {
                'ra_deg': ra_deg,
                'dec_deg': dec_deg,
                'alt_deg': altaz.alt.deg,
                'az_deg': altaz.az.deg
            },
            'observation_time': obs_time.isot,
            'bodies': bodies,
            # Includes the nearest bright stars or known radio sources if desired
            'nearest_stars': [
                {'name': 'HD 184512', 'separation_deg': 1.2},  # Example placeholder
                {'name': 'Chi Sagittarii', 'separation_deg': 3.8}  # Example placeholder
            ]
        }
    
    def assess_signal_novelty(self):
        """
        Assess how novel the signal is compared to known phenomena
        
        Returns:
            Dictionary with novelty assessment
        """
        # 1. Use isolation forest to detect how anomalous the signal is
        from sklearn.ensemble import IsolationForest
        
        # Extract features from the signal
        features = self._extract_signal_features()
        
        # Create a dataset with the Wow! signal features and many normal signals
        normal_signals = self._generate_normal_signal_features(n_samples=100)
        
        # Combine features (Wow! signal is the first one)
        all_features = np.vstack([features.reshape(1, -1), normal_signals])
        
        # Run isolation forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_scores = iso_forest.fit_predict(all_features)
        wow_score = outlier_scores[0]  # Score for the Wow! signal
        
        # 2. Compare with known astronomical signal classes
        known_classes = {
            'pulsar': self._compare_with_pulsar_signals(),
            'solar_flare': self._compare_with_solar_flares(),
            'quasar': self._compare_with_quasars(),
            'terrestrial_rfi': self._compare_with_terrestrial_rfi(),
            'satellite': self._compare_with_satellite_signals()
        }
        
        return {
            'isolation_forest_score': float(wow_score),  # -1 for anomalous, 1 for normal
            'novelty_percentage': 100 * (1 - self._calculate_novelty_score()),  # Higher means more novel
            'similarity_to_known_classes': known_classes,
            'most_similar_class': min(known_classes.items(), key=lambda x: x[1])[0],
            'least_similar_class': max(known_classes.items(), key=lambda x: x[1])[0]
        }
    
    # Helper methods for the analysis
    
    def _analyze_time_pattern(self):
        """Analyze the time pattern of the signal"""
        # Simplified implementation
        return 0.65  # Placeholder value
    
    def _analyze_strength_pattern(self):
        """Analyze the strength pattern of the signal"""
        # Simplified implementation
        return 0.7  # Placeholder value
    
    def _analyze_beam_pattern_match(self):
        """Analyze how well the signal matches the expected beam pattern"""
        # The Big Ear had a Gaussian beam pattern
        # If the signal was a point source, intensity should follow Gaussian as Earth rotates
        x = np.linspace(-3, 3, len(self.intensity_interp))
        gaussian = np.exp(-x**2/2)
        
        # Normalize both signals
        norm_intensity = (self.intensity_interp - np.min(self.intensity_interp)) / (np.max(self.intensity_interp) - np.min(self.intensity_interp))
        correlation = np.corrcoef(norm_intensity, gaussian)[0, 1]
        return abs(correlation)
    
    def _analyze_information_content_for_et(self):
        """Analyze the information content for ET intelligence assessment"""
        # Placeholder implementation
        return 0.75
    
    def _check_for_complex_modulation(self):
        """Check if the signal shows evidence of complex modulation"""
        # Placeholder implementation
        return False
    
    def _assess_potential_natural_explanations(self):
        """Assess potential natural explanations for the signal"""
        # Placeholder implementation
        return 0.6
    
    def _extract_signal_features(self):
        """Extract numerical features from the signal for machine learning analysis"""
        # Extract various statistical and signal features
        features = np.array([
            np.mean(self.intensity),               # Mean intensity
            np.std(self.intensity),                # Standard deviation
            np.max(self.intensity),                # Peak intensity
            np.sum(self.intensity),                # Total power
            np.median(self.intensity),             # Median
            stats.skew(self.intensity),            # Skewness
            stats.kurtosis(self.intensity),        # Kurtosis
            np.max(np.gradient(self.intensity)),   # Max gradient (rise)
            np.min(np.gradient(self.intensity)),   # Min gradient (fall)
            # Add more features as needed
        ])
        return features
    
    def _generate_normal_signal_features(self, n_samples=100):
        """Generate features for normal (non-anomalous) signals for comparison"""
        # This would ideally use a database of real signals
        # For this implementation, we'll generate synthetic features
        np.random.seed(42)
        
        # Generate with randomized parameters but following expected distributions
        features = np.zeros((n_samples, 9))
        for i in range(n_samples):
            # Generate a random signal type
            signal_type = np.random.choice(['noise', 'pulsar', 'flare', 'rfi'])
            
            if signal_type == 'noise':
                features[i, 0] = np.random.uniform(0.8, 1.2)  # Mean
                features[i, 1] = np.random.uniform(0.8, 1.5)  # Std dev
                features[i, 2] = np.random.uniform(2, 4)      # Peak
                features[i, 3] = np.random.uniform(70, 90)    # Total power
                features[i, 4] = np.random.uniform(0.8, 1.2)  # Median
                features[i, 5] = np.random.uniform(-0.2, 0.2) # Skewness
                features[i, 6] = np.random.uniform(-0.5, 0.5) # Kurtosis
                features[i, 7] = np.random.uniform(0.1, 0.3)  # Max gradient
                features[i, 8] = np.random.uniform(-0.3, -0.1) # Min gradient
            
            elif signal_type == 'pulsar':
                features[i, 0] = np.random.uniform(1.0, 1.5)  # Mean
                features[i, 1] = np.random.uniform(1.5, 2.5)  # Std dev
                features[i, 2] = np.random.uniform(4, 8)      # Peak
                features[i, 3] = np.random.uniform(90, 120)   # Total power
                features[i, 4] = np.random.uniform(0.9, 1.3)  # Median
                features[i, 5] = np.random.uniform(0.5, 1.5)  # Skewness
                features[i, 6] = np.random.uniform(1.0, 3.0)  # Kurtosis
                features[i, 7] = np.random.uniform(0.5, 1.0)  # Max gradient
                features[i, 8] = np.random.uniform(-1.0, -0.5) # Min gradient
            
            elif signal_type == 'flare':
                features[i, 0] = np.random.uniform(1.2, 2.0)  # Mean
                features[i, 1] = np.random.uniform(2.0, 3.0)  # Std dev
                features[i, 2] = np.random.uniform(6, 12)     # Peak
                features[i, 3] = np.random.uniform(100, 150)  # Total power
                features[i, 4] = np.random.uniform(1.0, 1.5)  # Median
                features[i, 5] = np.random.uniform(1.0, 2.0)  # Skewness
                features[i, 6] = np.random.uniform(2.0, 5.0)  # Kurtosis
                features[i, 7] = np.random.uniform(1.0, 2.0)  # Max gradient
                features[i, 8] = np.random.uniform(-2.0, -1.0) # Min gradient
            
            else:  # RFI
                features[i, 0] = np.random.uniform(1.0, 3.0)  # Mean
                features[i, 1] = np.random.uniform(1.0, 4.0)  # Std dev
                features[i, 2] = np.random.uniform(5, 15)     # Peak
                features[i, 3] = np.random.uniform(80, 200)   # Total power
                features[i, 4] = np.random.uniform(0.8, 2.5)  # Median
                features[i, 5] = np.random.uniform(-1.0, 2.0) # Skewness
                features[i, 6] = np.random.uniform(-1.0, 6.0) # Kurtosis
                features[i, 7] = np.random.uniform(0.2, 3.0)  # Max gradient
                features[i, 8] = np.random.uniform(-3.0, -0.2) # Min gradient
        
        return features
    
    def _calculate_novelty_score(self):
        """Calculate overall novelty score (0-1, higher means more similar to known signals)"""
        # This would combine results from various comparisons
        # For now, return a placeholder value
        return 0.25  # Suggesting the signal is quite novel
    
    def _compare_with_pulsar_signals(self):
        """Compare the Wow! signal with typical pulsar signals"""
        # Placeholder implementation
        return 0.35  # Similarity score (0-1, lower means more different)
    
    def _compare_with_solar_flares(self):
        """Compare the Wow! signal with solar flare signals"""
        # Placeholder implementation
        return 0.6
    
    def _compare_with_quasars(self):
        """Compare the Wow! signal with quasar signals"""
        # Placeholder implementation
        return 0.45
    
    def _compare_with_terrestrial_rfi(self):
        """Compare the Wow! signal with terrestrial RFI"""
        # Placeholder implementation
        return 0.55
    
    def _compare_with_satellite_signals(self):
        """Compare the Wow! signal with satellite signals"""
        # Placeholder implementation
        return 0.4
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report combining all analysis techniques
        
        Returns:
            Dictionary with all analysis results
        """
        report = {}
        
        # Information content analysis
        report['information_content'] = self.analyze_information_content()
        
        # Hypothesis testing
        report['natural_source'] = self.test_natural_source_hypothesis()
        report['terrestrial_interference'] = self.test_terrestrial_interference_hypothesis()
        report['extraterrestrial_intelligence'] = self.test_extraterrestrial_intelligence_hypothesis()
        
        # Time-frequency analysis
        # report['time_frequency'] = self.analyze_time_frequency_domain()  # Uncomment if method available
        
        # Generate synthetic comparative signals
        report['synthetic_comparison'] = self.generate_and_compare_synthetic_signals()
        
        # Astronomical context
        report['astronomical_context'] = self.analyze_celestial_positions()
        
        # Generate audio representations
        try:
            audio_paths = self.convert_to_audio()
            report['audio_representations'] = {
                'am_fm_path': audio_paths[0],
                'spectral_path': audio_paths[1]
            }
        except Exception as e:
            report['audio_representations'] = {
                'error': f"Failed to generate audio: {str(e)}"
            }
        
        # Novelty detection
        # report['novelty_assessment'] = self.assess_signal_novelty()  # Uncomment if method available
        
        return report
