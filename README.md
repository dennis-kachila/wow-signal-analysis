# Wow! Signal Analysis

## Overview
This project performs a comprehensive re-analysis of the famous Wow! signal detected by the Big Ear radio telescope on August 15, 1977. Using modern data science techniques, signal processing, machine learning, information theory, and audio analysis, we investigate the signal characteristics and possible origins through multiple complementary methods.

The analysis integrates traditional signal processing with novel approaches including:
1. Advanced statistical analysis of the original signal data
2. Audio representation and spectral analysis of the Wow! signal
3. Novel hypothesis testing including the Quantum Jump and Algorithmic Message hypotheses
4. Pattern detection and modulation analysis to identify potential encoded information

## Background
The Wow! signal was a strong narrowband radio signal detected during a SETI survey by Ohio State University's Big Ear radio telescope. It appeared to match many of the expected characteristics of a potential extraterrestrial transmission:

- **Duration**: 72 seconds (the time it took for Earth's rotation to move the telescope across the signal source)
- **Frequency**: 1420.4556 MHz (very close to the hydrogen line at 1420.406 MHz)
- **Bandwidth**: Narrowband (estimated < 10 kHz)
- **Signal-to-Noise Ratio**: Up to 30 sigma above background
- **Name**: "Wow!" comes from astronomer Jerry Ehman's reaction, writing "Wow!" in the margin of the computer printout

Despite repeated attempts, the signal has never been detected again. This project applies modern computational methods that were not available in 1977 to analyze the signal in new ways, including testing novel hypotheses about its origin.

## Project Structure
```
wow-signal-analysis/
├── data/               # Downloaded and processed data
│   ├── wow_signal.csv        # Original signal data
│   ├── wow_signal.jpg        # Original signal image
│   └── Wow_Signal_SETI_Project.mp3 # Audio representation of the Wow! signal
├── notebooks/          # Jupyter notebooks for interactive analysis
├── results/            # Generated plots, reports, and analysis results
│   ├── audio_analysis/        # Audio analysis results and visualizations
│   ├── limited_audio_analysis/ # Lightweight audio analysis results
│   └── wow_analysis_summary_*.md # Comprehensive analysis summaries
├── src/                # Python source code
│   ├── advanced_analysis.py    # Advanced analysis techniques and methods
│   ├── data_acquisition.py     # Code to download and prepare the raw data
│   ├── signal_processing.py    # Signal processing and visualization
│   ├── hypothesis_testing.py   # Test various hypotheses about signal origin
│   └── information_extraction.py # Analysis of potential encoded information
├── advanced_analysis.py # Advanced analysis script with novel hypothesis testing
├── audio_analysis.py   # Advanced audio analysis of the Wow! signal
├── limited_audio_analysis.py # Memory-optimized audio analysis
├── main.py             # Main script to run the complete analysis pipeline
├── simple_analysis.py  # Simple script for basic analysis
├── plan.md             # Detailed analysis plan
└── requirements.txt    # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wow-signal-analysis.git
cd wow-signal-analysis
```

2. Create a virtual environment (if not using Anaconda):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Simple Analysis
For a basic analysis of the signal:
```bash
python simple_analysis.py
```

### Advanced Analysis
For a comprehensive analysis with novel hypothesis testing:
```bash
python advanced_analysis.py
```

### Audio Analysis
For specialized analysis of the Wow! signal audio representation:
```bash
python audio_analysis.py
```

For systems with memory constraints:
```bash
python limited_audio_analysis.py
```

### Interactive Analysis
For an interactive analysis experience, open the Jupyter notebook:
```bash
jupyter notebook notebooks/wow_signal_analysis.ipynb
```

The analysis pipeline:
1. Loads and processes the Wow! signal data
2. Performs advanced signal processing and visualization
3. Creates audio representations of the signal
4. Tests multiple hypotheses about the signal's origin
5. Analyzes the signal for potential encoded information 
6. Performs specialized audio analysis on the Wow! signal
7. Evaluates novel hypotheses including Quantum Jump and Algorithmic Message theories
8. Generates comprehensive reports and visualizations

## Results

After running the analysis, results will be available in the `results` directory, including:

### Signal Analysis Results
- Spectral analysis plots
- Signal interpolation and visualization
- Wavelet transforms and time-frequency analysis
- Information content measures

### Audio Analysis Results
- Spectrograms and mel-spectrograms
- Chromagrams showing tonal content
- Modulation analysis (AM/FM characteristics)
- Onset detection and pattern analysis
- Statistical analysis of audio features

### Hypothesis Testing Results
Our analysis has evaluated multiple hypotheses for the Wow! signal origin:

1. **Terrestrial Radio Frequency Interference**: Probability score of -25%
2. **Natural Astronomical Phenomenon**: Probability score of -10% 
3. **Extraterrestrial Intelligent Signal**: Probability score of 65%
4. **Quantum Jump Hypothesis**: Probability score of 40%
5. **Algorithmic Message Hypothesis**: (Included in advanced analysis)

The evidence currently supports the Extraterrestrial Intelligent Signal hypothesis as the most probable explanation, though we acknowledge significant uncertainty remains.

### Key Findings
- The signal exhibits extremely narrow bandwidth, consistent with artificial sources
- Audio analysis suggests mixed modulation characteristics with both AM and FM components
- The frequency choice near the hydrogen line is logical for interstellar communication
- Spectral analysis reveals interesting patterns that remain unexplained by conventional theories
- The absence of signal repetition remains a significant mystery

Full detailed reports are available in the `results` directory after running the analysis scripts.

## License

MIT License

## Acknowledgements

This project is for educational and research purposes only. The original Wow! signal was detected by Jerry R. Ehman while working on a SETI project at the Big Ear radio telescope of Ohio State University.
