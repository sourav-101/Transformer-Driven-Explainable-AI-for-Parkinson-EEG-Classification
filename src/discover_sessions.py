from mne_bids import BIDSPath, get_entity_vals
from config import RAW_DIR

def get_all_recordings():
    """
    Returns a list of dicts: {subject, session, bids_path}
    session is 'off', 'on', or None (for HC, which has no session split).
    """
    subjects = get_entity_vals(RAW_DIR, "subject")
    sessions = get_entity_vals(RAW_DIR, "session")  

    recordings = []
    for sub in subjects:
        found_session = False
        for ses in sessions:
            bp = BIDSPath(subject=sub, session=ses, task="rest",
                           datatype="eeg", root=RAW_DIR)
            if bp.fpath.exists():
                recordings.append({"subject": sub, "session": ses, "bids_path": bp})
                found_session = True
        if not found_session:
            # HC subjects typically have no session entity
            bp = BIDSPath(subject=sub, task="rest", datatype="eeg", root=RAW_DIR)
            if bp.fpath.exists():
                recordings.append({"subject": sub, "session": None, "bids_path": bp})

    return recordings

if __name__ == "__main__":
    recs = get_all_recordings()
    print(f"Found {len(recs)} recordings")
    for r in recs[:5]:
        print(r["subject"], r["session"])