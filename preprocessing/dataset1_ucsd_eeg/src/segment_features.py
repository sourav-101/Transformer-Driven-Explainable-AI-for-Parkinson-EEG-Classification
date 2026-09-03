import json
import numpy as np
import mne
import torch
from itertools import combinations
from scipy.signal import hilbert
from config import (
    CROP_SECONDS, EPOCH_SECONDS, SAMPLING_RATE,
    FREQ_BANDS, CHANNELS_32, PROCESSED_DIR, INTERIM_DIR,
)

def compute_band_power(epoch_data, sfreq, fmin, fmax):
    """epoch_data: (n_channels, n_samples). Returns (n_channels,) power via DFT/PSD."""
    n = epoch_data.shape[1]
    freqs = np.fft.rfftfreq(n, d=1 / sfreq)
    fft_vals = np.fft.rfft(epoch_data, axis=1)
    psd = (np.abs(fft_vals) ** 2) / (2 * np.pi * n)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, band_mask].mean(axis=1)

def compute_plv_from_phase(phase_epoch):
    """phase_epoch: (n_channels, n_samples) instantaneous phase for one epoch,
    already band-filtered over the FULL recording before epoching.
    Returns (n_pairs,) PLV for this epoch."""
    n_ch = phase_epoch.shape[0]
    plvs = []
    for i, j in combinations(range(n_ch), 2):
        diff = phase_epoch[i] - phase_epoch[j]
        plv = np.abs(np.mean(np.exp(1j * diff)))
        plvs.append(plv)
    return np.array(plvs)


def infer_group(subject):
    """PD subjects are named like 'pd12', HC subjects like 'hc10'.
    Explicit, so downstream code never has to re-parse the subject string."""
    s = subject.lower()
    if s.startswith("pd"):
        return "PD"
    if s.startswith("hc"):
        return "HC"
    raise ValueError(f"Cannot infer group from subject id: {subject!r}")


def load_ica_log(subject, session):
    """Read back the ICA QC info preprocess.py wrote out, so it isn't lost
    between stages. Returns -1 (with a warning) if the log is missing, so an
    old/incomplete run doesn't hard-crash feature extraction."""
    ses_tag = session if session else "hc"
    log_path = INTERIM_DIR / f"sub-{subject}_ses-{ses_tag}_ica_log.json"
    if not log_path.exists():
        print(f"[{subject}-{ses_tag}] WARNING: no ICA log found at {log_path}, "
              f"n_ica_dropped will be recorded as -1")
        return -1
    with open(log_path) as f:
        return json.load(f)["n_ica_dropped"]


def process_file(fif_path, subject, session):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    if raw.times[-1] < CROP_SECONDS:
        print(f"[{subject}-{session or 'hc'}] SKIPPED: recording is only "
              f"{raw.times[-1]:.1f}s long, shorter than the required "
              f"{CROP_SECONDS:.0f}s crop window.")
        return None

    # Crop to first 180 s
    raw.crop(tmin=0, tmax=CROP_SECONDS, include_tmax=False)
    data = raw.get_data()  # (n_channels, n_samples) -- full continuous 180s

    ch_names = raw.ch_names
    n_channels_actual = len(ch_names)
    if n_channels_actual != len(CHANNELS_32):
        print(f"[{subject}-{session or 'hc'}] NOTE: this file has "
              f"{n_channels_actual} channels, not the nominal {len(CHANNELS_32)} "
              f"-- saved ch_names/channel_pairs reflect the actual channels.")

    samples_per_epoch = int(EPOCH_SECONDS * sfreq)
    n_epochs = data.shape[1] // samples_per_epoch

    # --- Pre-compute band-filtered continuous data + Hilbert phase ONCE per band,
    # over the full 180s signal -- NOT per 1-second epoch. This avoids the
    # "filter longer than signal" distortion entirely, since the filter kernel
    # (up to ~1691 samples for delta) is now much shorter than the 180s/92160-sample
    # signal it's applied to.
    band_names = list(FREQ_BANDS.keys())
    band_phases = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
        band_phases[band_name] = np.angle(hilbert(filt, axis=1))  # (n_channels, n_samples)

    psd_features, plv_features = [], []
    for e in range(n_epochs):
        start, end = e * samples_per_epoch, (e + 1) * samples_per_epoch
        seg = data[:, start:end]

        # PSD: DFT per epoch is fine as-is -- this was never a "filter", just an FFT
        # on a fixed-length window, so no filter-length distortion applies here.
        # Kept band-major: shape (n_bands, n_channels) per epoch.
        psd_bands = np.stack(
            [compute_band_power(seg, sfreq, lo, hi) for lo, hi in FREQ_BANDS.values()]
        )  # (n_bands, n_channels)

        # PLV: slice the pre-computed continuous phase per band for this epoch's window.
        # Kept band-major: shape (n_bands, n_pairs) per epoch.
        plv_bands = np.stack(
            [compute_plv_from_phase(band_phases[bn][:, start:end]) for bn in band_names]
        )  # (n_bands, n_pairs)

        psd_features.append(psd_bands)
        plv_features.append(plv_bands)

    # --- Shape as (n_epochs, n_bands, n_channels/n_pairs) rather than a flat
    # concatenated vector. This is the whole point of the change: with a flat
    # vector, "give me beta power" means knowing the code's internal band
    # ordering and slicing psd[:, 96:128] by hand -- a silent, easy-to-get-wrong
    # dependency on this file's implementation. With this shape,
    # `psd_arr[:, band_names.index("beta"), :]` is unambiguous and self-describing,
    # and matches up 1:1 with the `bands` list saved alongside it below.
    psd_arr = np.stack(psd_features)  # (n_epochs, n_bands, n_channels)
    plv_arr = np.stack(plv_features)  # (n_epochs, n_bands, n_pairs)

    # --- NOTE: features are saved RAW (unnormalized) on purpose.
    # Normalization is deferred to training time, computed per LOSO fold from
    # the training subjects only, then applied unchanged to the held-out test
    # subject. Baking a per-recording z-score in here would zero out each
    # subject's own baseline power/connectivity level -- exactly the kind of
    # group-level difference (e.g. reduced beta/gamma in PD) the paper's own
    # background section says matters. See normalize.py for the fold-aware
    # normalization helper used at training time.

    ses_tag = session if session else "hc"
    group = infer_group(subject)
    n_ica_dropped = load_ica_log(subject, session)

    # Explicit channel-pair labels for the PLV axis, in the same order
    # `compute_plv_from_phase` produces them (itertools.combinations over
    # this file's actual channel list). Without this, the pair identity
    # behind PLV column k is only recoverable by re-deriving the combinations
    # by hand -- and would be WRONG for any subject with a missing channel
    # if derived from the nominal CHANNELS_32 list instead of ch_names.
    channel_pairs = list(combinations(ch_names, 2))  # list of (ch_i, ch_j) tuples

    out = {
        "psd": torch.tensor(psd_arr, dtype=torch.float32),   # (n_epochs, n_bands, n_channels)
        "plv": torch.tensor(plv_arr, dtype=torch.float32),   # (n_epochs, n_bands, n_pairs)
        "subject": subject,
        "session": ses_tag,
        "group": group,
        "n_epochs": n_epochs,
        "n_ica_dropped": n_ica_dropped,
        "ch_names": ch_names,               # this file's ACTUAL channels, not a nominal constant
        "bands": band_names,                # index i of the band axis == bands[i]
        "channel_pairs": channel_pairs,     # index k of the PLV pair axis == channel_pairs[k]
        "normalized": False,                # explicit flag: these are raw features
        "sfreq": float(sfreq),
    }
    out_path = PROCESSED_DIR / f"sub-{subject}_ses-{ses_tag}_features.pt"
    torch.save(out, out_path)
    print(f"Saved {out_path} | PSD {tuple(psd_arr.shape)}, PLV {tuple(plv_arr.shape)}, "
          f"group={group}, n_ica_dropped={n_ica_dropped}")
    return out_path


if __name__ == "__main__":
    from discover_sessions import get_all_recordings
    recs = get_all_recordings()
    for r in recs:
        ses_tag = r["session"] if r["session"] else "hc"
        fif_path = INTERIM_DIR / f"sub-{r['subject']}_ses-{ses_tag}_cleaned_raw.fif"
        if fif_path.exists():
            process_file(fif_path, r["subject"], r["session"])
        else:
            print(f"Missing interim file for {r['subject']}-{ses_tag}, skipping")