# Paths
import os

FOLDER_PATH = "./OBE_Input"                 # Directory containing .OBE EEG files
CSV_PATH = "./EEG_data_sheet.csv"           # Seizure metadata spreadsheet

# Everything lands under one root, grouped by analysis then by scope (per-seizure / per-animal / global).
OUTPUT_ROOT = "Output"

OVERVIEW_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "Overview")                  # 60-minute overview plots

# Per-seizure figures from the main analysis loop.
CORRELATION_SEIZURE_DIR = os.path.join(OUTPUT_ROOT, "Correlation", "Per_Seizure")
COHERENCE_SEIZURE_DIR = os.path.join(OUTPUT_ROOT, "Coherence", "Per_Seizure")
PHASE_DELAY_SEIZURE_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Per_Seizure")

# Per-animal longitudinal summaries.
CORRELATION_ANIMAL_DIR = os.path.join(OUTPUT_ROOT, "Correlation", "Per_Animal")
COHERENCE_ANIMAL_DIR = os.path.join(OUTPUT_ROOT, "Coherence", "Per_Animal")
PHASE_DELAY_ANIMAL_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Per_Animal")

# Global, Day-0-normalized summaries across all animals.
CORRELATION_GLOBAL_DIR = os.path.join(OUTPUT_ROOT, "Correlation", "Global")
COHERENCE_GLOBAL_DIR = os.path.join(OUTPUT_ROOT, "Coherence", "Global")
PHASE_DELAY_GLOBAL_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Global")

# Exploratory phase-delay grids (raw MSC, phase-vs-frequency).
PHASE_DELAY_EXPLORATION_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Exploration")

# Correlation parameters
CORR_THRESHOLD = 0.65          # |correlation| above this counts as "synchronized"
ROLLING_WINDOW_SECS = 1.0      # Rolling correlation window length (seconds)
ROLLING_STEP_SECS = 0.5        # Rolling correlation step / hop size (seconds)
PADDING_WINDOW_SECS = 120.0    # Half-width of the window centered on each seizure

# Coherence parameters
COH_WINDOW = 256               # Welch segment / window length (samples)
COH_NOVERLAP = 128             # Overlap between adjacent Welch segments (samples)
COH_NFFT = 1024                # FFT length -> frequency resolution
COH_BAND_LOW_HZ = (0.0, 14.0)  # Low-frequency band of interest (Hz)
COH_BAND_HIGH_HZ = (14.0, 55.0)  # High-frequency band of interest (Hz)
PRE_SEIZURE_SECS = 60          # Pre-seizure window used for the coherence baseline
PHASE_MSC_THRESHOLD = 0.1   # MSC gate for phase slope fitting (0-14 Hz band)

# Baseline filtering
BASELINE_DURATION_SECS = 120   # Window before seizure start used for baseline correlation
BASELINE_CORR_THRESHOLD = 0.5  # Exclude seizures whose baseline |correlation| exceeds this

# Peak filtering (artificial bias); kept even when currently toggled off.
USE_INTRA_SEIZURE_TOP = True   # If True, keep only the top fraction of intra-seizure values
INTRA_PERCENT_VAL = 0.10       # Fraction kept when USE_INTRA_SEIZURE_TOP is enabled

# EEG channel configuration
# Channel index = (recording_order - 1) * CHANNEL_STRIDE + offset  (1-based)
CHANNEL_STRIDE = 10            # Channels per recording order
CH1_OFFSET = 5                 # Offset for channel 1
CH2_OFFSET = 9                 # Offset for channel 2

# Matching
DST_OFFSET_HOURS = 1           # Daylight-savings correction subtracted from header time

# Plotting configuration
DOWNSAMPLE_FACTOR = 10         # Decimation factor for overview plots
NUM_MINS_PER_SUBPLOT = 10      # Minutes shown per overview subplot
OVERVIEW_NUM_SUBPLOTS = 6      # Subplots per overview figure (-> 60 min total)
SEGMENT_SECS = 10.0            # Seconds per zoomed overlap segment
COLORMAP = "viridis"           # Matplotlib colormap for Racine-coded scatter points
RACINE_SCALE_MAX = 7           # Racine severity scale upper bound
DAY_ZERO = (2000, 1, 1)        # Shared epoch (Y, M, D) for Day-0 normalized timelines
