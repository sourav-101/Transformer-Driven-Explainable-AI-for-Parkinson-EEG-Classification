from discover_sessions import get_all_recordings
from preprocess import preprocess_recording
from segment_features import process_file


def main():
    recs = get_all_recordings()
    print(f"Processing {len(recs)} recordings...")

    succeeded, failed = [], []

    for r in recs:
        subject, session = r["subject"], r["session"]
        tag = f"{subject}-{session or 'hc'}"
        try:
            fif_path = preprocess_recording(r["bids_path"], subject, session)
            if fif_path is None:
                # preprocess_recording already printed the reason (e.g. ICA fit
                # failure). Don't call process_file on a None path -- record
                # it as failed and move on to the next recording.
                print(f"[{tag}] SKIPPED (preprocessing failed, see above)")
                failed.append(tag)
                continue

            result_path = process_file(fif_path, subject, session)
            if result_path is None:
                print(f"[{tag}] SKIPPED (segment_features: recording too short, see above)")
                failed.append(tag)
                continue

            succeeded.append(tag)

        except Exception as e:
            # Catch-all: anything unexpected in EITHER stage (a corrupt file,
            # a shape mismatch, an I/O error, etc.) is logged and the batch
            # continues, instead of one bad recording killing the whole run.
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            failed.append(tag)

    print("\n" + "=" * 50)
    print(f"Done. {len(succeeded)}/{len(recs)} recordings succeeded.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
    print("=" * 50)


if __name__ == "__main__":
    main()