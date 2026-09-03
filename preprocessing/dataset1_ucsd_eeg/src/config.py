from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # BIDS root (ds002778 downloaded here)
INTERIM_DIR = ROOT / "data" / "interim"   # filtered + ICA-cleaned continuous data
PROCESSED_DIR = ROOT / "data" / "processed"  # final PSD/PLV tensors

for d in [INTERIM_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Paper's 32 channels (UC dataset config) ---
CHANNELS_32 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5",
    "P7", "P3", "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8",
    "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2",
    "Fz", "Cz",
]

# --- Filtering ---
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 50.0
NOTCH_LOW = 48.0
NOTCH_HIGH = 52.0   

# --- Segmentation ---
CROP_SECONDS = 180.0   # first 3 minutes
EPOCH_SECONDS = 1.0    # 1-second time samples
SAMPLING_RATE = 512    # native rate for ds002778

# --- Frequency bands for PSD/PLV ---
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 48),
}