import os

FOLDER_PATH = "./OBE_Input"
CSV_PATH = "./EEG_data_sheet.csv"

# explainer: everything lands under one root, by analysis then scope (per seizure, per animal, global)
OUTPUT_ROOT = "Output"

OVERVIEW_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "Overview")

CORRELATION_SEIZURE_DIR = os.path.join(OUTPUT_ROOT, "Correlation", "Per_Seizure")
COHERENCE_SEIZURE_DIR = os.path.join(OUTPUT_ROOT, "Coherence", "Per_Seizure")
PHASE_DELAY_SEIZURE_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Per_Seizure")

CORRELATION_ANIMAL_DIR = os.path.join(OUTPUT_ROOT, "Correlation", "Per_Animal")
COHERENCE_ANIMAL_DIR = os.path.join(OUTPUT_ROOT, "Coherence", "Per_Animal")
PHASE_DELAY_ANIMAL_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Per_Animal")

CORRELATION_GLOBAL_DIR = os.path.join(OUTPUT_ROOT, "Correlation", "Global")
COHERENCE_GLOBAL_DIR = os.path.join(OUTPUT_ROOT, "Coherence", "Global")
PHASE_DELAY_GLOBAL_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Global")

PHASE_DELAY_EXPLORATION_DIR = os.path.join(OUTPUT_ROOT, "Phase_Delay", "Exploration")

CORR_THRESHOLD = 0.65  # decision: |r| above this counts as synchronized
ROLLING_WINDOW_SECS = 1.0  # explainer: seconds
ROLLING_STEP_SECS = 0.5  # explainer: hop, seconds
PADDING_WINDOW_SECS = 120.0  # explainer: half-width of the window around each seizure

COH_WINDOW = 256  # explainer: Welch segment length, samples
COH_NOVERLAP = 128  # explainer: samples
COH_NFFT = 1024  # explainer: sets the frequency resolution
COH_BAND_LOW_HZ = (0.0, 14.0)  # explainer: Hz
COH_BAND_HIGH_HZ = (14.0, 55.0)  # explainer: Hz
PRE_SEIZURE_SECS = 60  # decision: pre-seizure window used as the coherence baseline
PHASE_MSC_THRESHOLD = 0.1  # decision: MSC gate for the phase-slope fit

BASELINE_DURATION_SECS = 120  # explainer: window before onset used for the baseline correlation
BASELINE_CORR_THRESHOLD = 0.5  # decision: drop seizures whose baseline |r| is already above this

# decision: peak filtering biases the result, kept here but off
USE_INTRA_SEIZURE_TOP = True  # explainer: keep only the top fraction of intra-seizure values
INTRA_PERCENT_VAL = 0.10

# explainer: channel = (recording_order - 1) * CHANNEL_STRIDE + offset, 1-based
CHANNEL_STRIDE = 10  # explainer: channels per recording order
CH1_OFFSET = 5
CH2_OFFSET = 9

DST_OFFSET_HOURS = 1  # decision: header times are off by daylight savings, subtract this

DOWNSAMPLE_FACTOR = 10
NUM_MINS_PER_SUBPLOT = 10
OVERVIEW_NUM_SUBPLOTS = 6  # explainer: x NUM_MINS_PER_SUBPLOT = 60 min
SEGMENT_SECS = 10.0  # explainer: seconds per zoomed segment
COLORMAP = "viridis"
RACINE_SCALE_MAX = 7
DAY_ZERO = (2000, 1, 1)  # decision: shared Day 0 so all animals line up on one timeline
