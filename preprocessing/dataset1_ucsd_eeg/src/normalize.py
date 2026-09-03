"""
Training-time normalization for MCPNet-style LOSO experiments.

Why this lives here instead of in segment_features.py:
segment_features.py now saves RAW psd/plv tensors, shaped
(n_epochs, n_bands, n_channels_or_pairs). Normalization statistics (mean/std)
are computed per LOSO fold from the training subjects ONLY, then applied
unchanged to the held-out test subject. This avoids two problems with
normalizing inside segment_features.py:

  1. Leakage risk if you ever change the normalization scope later (e.g. to
     pool across subjects) without noticing it now includes the test subject.
  2. Per-recording normalization erasing between-subject group differences
     (e.g. PD patients having systematically lower beta/gamma power) --
     computing stats from the training pool instead keeps that signal intact
     for the model to learn from, while still not leaking test-subject info
     into the transform.
"""

from pathlib import Path
import torch
from config import PROCESSED_DIR


def load_subject_features(subject, session):
    """Load one subject's raw feature file saved by segment_features.py."""
    ses_tag = session if session else "hc"
    path = PROCESSED_DIR / f"sub-{subject}_ses-{ses_tag}_features.pt"
    return torch.load(path)


def fit_normalizer(train_records):
    """
    train_records: list of feature dicts (as returned by load_subject_features)
    for every subject in the TRAINING side of one LOSO fold. Excludes the
    held-out test pair.

    Returns a dict of {psd_mean, psd_std, plv_mean, plv_std}, each shaped
    (n_bands, n_channels_or_pairs) so it broadcasts against a (n_epochs,
    n_bands, n_channels_or_pairs) tensor.
    """
    all_psd = torch.cat([r["psd"] for r in train_records], dim=0)  # (E_total, bands, ch)
    all_plv = torch.cat([r["plv"] for r in train_records], dim=0)  # (E_total, bands, pairs)

    eps_psd = 1e-8 * all_psd.abs().max()
    eps_plv = 1e-8

    return {
        "psd_mean": all_psd.mean(dim=0),           # (bands, ch)
        "psd_std": all_psd.std(dim=0) + eps_psd,    # (bands, ch)
        "plv_mean": all_plv.mean(dim=0),            # (bands, pairs)
        "plv_std": all_plv.std(dim=0) + eps_plv,    # (bands, pairs)
    }


def apply_normalizer(record, stats):
    """
    Apply previously-fit train-set statistics to ANY record (train or test).
    Returns a new dict; does not mutate the input.
    """
    out = dict(record)
    out["psd"] = (record["psd"] - stats["psd_mean"]) / stats["psd_std"]
    out["plv"] = (record["plv"] - stats["plv_mean"]) / stats["plv_std"]
    out["normalized"] = True
    return out


def normalize_loso_fold(train_records, test_record):
    """
    Convenience wrapper for one LOSO iteration.

    train_records: list of raw feature dicts for the training subjects
    test_record:   raw feature dict for the held-out subject

    Returns (train_records_normalized, test_record_normalized), where the
    test subject's own values never influenced the mean/std used to
    normalize it.
    """
    stats = fit_normalizer(train_records)
    train_norm = [apply_normalizer(r, stats) for r in train_records]
    test_norm = apply_normalizer(test_record, stats)
    return train_norm, test_norm


def get_band(record, band_name):
    """
    Pull out a single frequency band by name instead of by a hardcoded index.
    Works on either 'psd' or the record dict itself for both psd/plv.

    Example:
        rec = load_subject_features("pd6", "off")
        beta_psd = get_band(rec, "beta")["psd"]   # (n_epochs, n_channels)
        beta_plv = get_band(rec, "beta")["plv"]   # (n_epochs, n_pairs)
    """
    band_idx = record["bands"].index(band_name)
    return {
        "psd": record["psd"][:, band_idx, :],   # (n_epochs, n_channels)
        "plv": record["plv"][:, band_idx, :],   # (n_epochs, n_pairs)
    }


if __name__ == "__main__":
    # Minimal smoke test / usage example.
    from discover_sessions import get_all_recordings

    recs = get_all_recordings()
    loaded = []
    for r in recs:
        try:
            loaded.append(load_subject_features(r["subject"], r["session"]))
        except FileNotFoundError:
            continue

    if len(loaded) >= 2:
        test_record = loaded[0]
        train_records = loaded[1:]
        train_norm, test_norm = normalize_loso_fold(train_records, test_record)
        print(f"Fold example: {len(train_norm)} training subjects, "
              f"test subject = {test_norm['subject']}-{test_norm['session']}")

        beta = get_band(test_norm, "beta")
        print(f"Beta-band PSD for test subject: {tuple(beta['psd'].shape)} "
              f"(n_epochs, n_channels)")
        print(f"Beta-band PLV for test subject: {tuple(beta['plv'].shape)} "
              f"(n_epochs, n_pairs)")