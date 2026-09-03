import json
import mne
from mne_icalabel import label_components
from config import (
    CHANNELS_32, BANDPASS_LOW, BANDPASS_HIGH,
    NOTCH_LOW, NOTCH_HIGH, INTERIM_DIR,
)

def preprocess_recording(bids_path, subject, session):
    from mne_bids import read_raw_bids
    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()

    # 1. Channel selection (drop everything not in the paper's 32)
    available = [ch for ch in CHANNELS_32 if ch in raw.ch_names]
    missing = set(CHANNELS_32) - set(available)
    if missing:
        print(f"[{subject}-{session}] WARNING missing channels: {missing}")
    raw.pick_channels(available, ordered=True)

    # 1b. Common average reference — required for ICLabel, and standard
    # practice before ICA generally.
    raw.set_eeg_reference("average", projection=False)

    # 1c. Set montage early — needed for both ICA fitting and ICLabel.
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")

    # --- Branch 1: ICA fitting copy (1-100 Hz, fresh from the CAR'd raw,
    # BEFORE the paper's tight 0.5-50 filter is applied). This satisfies
    # ICLabel's expected input range and is generally better practice for
    # ICA decomposition stability.
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0, fir_design="firwin")

    n_components = len(raw_ica.ch_names) - 1  # -1 for the average-reference rank loss
    ica = mne.preprocessing.ICA(n_components=n_components, method="infomax",
                                 fit_params=dict(extended=True), random_state=42)

    ses_tag = session if session else "hc"
    try:
        ica.fit(raw_ica)
    except RuntimeError as e:
        print(f"[{subject}-{ses_tag}] ICA fit FAILED: {e}")
        return None

    labels = label_components(raw_ica, ica, method="iclabel")
    exclude_idx = [
        i for i, (label, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
        if label in ("eye blink", "heart beat", "channel noise")
        and prob > 0.7
    ]
    ica.exclude = exclude_idx
    print(f"[{subject}-{ses_tag}] excluding {len(exclude_idx)} ICA components: "
          f"{[labels['labels'][i] for i in exclude_idx]}")

    ica_log = {
        "subject": subject,
        "session": ses_tag,
        "n_ica_dropped": len(exclude_idx),
        "excluded_labels": [labels["labels"][i] for i in exclude_idx],
        "excluded_probs": [float(labels["y_pred_proba"][i]) for i in exclude_idx],
        "n_components": ica.n_components_,
    }
    log_path = INTERIM_DIR / f"sub-{subject}_ses-{ses_tag}_ica_log.json"
    with open(log_path, "w") as f:
        json.dump(ica_log, f, indent=2)

    # --- Branch 2: the REAL pipeline data, paper-exact filtering.
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, fir_design="firwin")

    notch_center = (NOTCH_LOW + NOTCH_HIGH) / 2  # 50 Hz
    notch_width = NOTCH_HIGH - NOTCH_LOW          # 4 Hz -> 48-52 band
    raw.notch_filter(freqs=[notch_center], notch_widths=notch_width,
                      trans_bandwidth=1.0, fir_design="firwin")

    # Apply the ICA weights (fit on raw_ica) to the paper-spec raw.
    ica.apply(raw)

    # 2. Save interim result
    out_path = INTERIM_DIR / f"sub-{subject}_ses-{ses_tag}_cleaned_raw.fif"
    raw.save(out_path, overwrite=True)
    return out_path


if __name__ == "__main__":
    from discover_sessions import get_all_recordings
    recs = get_all_recordings()
    for r in recs:
        preprocess_recording(r["bids_path"], r["subject"], r["session"])