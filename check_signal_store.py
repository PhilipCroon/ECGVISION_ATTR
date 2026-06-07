"""
Diagnostic: inspect the consolidated ECG signal store and its index, and figure
out the right fileID key to load post-2024 ECGs. Run on the Yale box:

    python check_signal_store.py

Answers three questions:
  1. Does qc_metadata.csv cover ECGs newer than the current key file
     (ecg_metadata_flagged_..._17july2025.csv)?  -> is there more data to pull?
  2. Which column in qc_metadata is the all_ecgs-relative path/key?
  3. Does that key actually resolve to a .npy on disk under all_ecgs/?
No writes, read-only.
"""
import os
import glob
import numpy as np
import pandas as pd

STORE      = '/mnt/raid0/bb2238/signals/preprocessed/all_ecgs'
QC         = os.path.join(STORE, 'qc_metadata.csv')
CUR_KEY    = '/mnt/nfs_yale_ecg_signals/data_LD/ecg_metadata_flagged_01jan2000_to_17july2025.csv'
CUTOFF     = '2025-07-17'   # current key file's coverage ceiling


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def show_dates(df, label):
    dc = [c for c in df.columns if 'date' in c.lower()]
    print(f"  [{label}] date cols: {dc}")
    for c in dc:
        s = pd.to_datetime(df[c], errors='coerce')
        after = int((s > CUTOFF).sum())
        print(f"    {c:30s} min={s.min()}  max={s.max()}  rows>{CUTOFF}={after}")
    return dc


def main():
    # --- 0. Is there a newer flagged key file anywhere obvious? ---
    section("0. Newer key files than the current one?")
    cur_mtime = os.path.getmtime(CUR_KEY) if os.path.exists(CUR_KEY) else 0
    print(f"  current key: {CUR_KEY}")
    print(f"    exists={os.path.exists(CUR_KEY)}  mtime={pd.to_datetime(cur_mtime, unit='s') if cur_mtime else 'n/a'}")
    for pat in [
        '/mnt/nfs_yale_ecg_signals/data_LD/ecg_metadata_flagged_*.csv',
        '/mnt/nfs_yale_ecg_signals/data_LD/*metadata*.csv',
    ]:
        for f in sorted(glob.glob(pat)):
            mt = pd.to_datetime(os.path.getmtime(f), unit='s')
            newer = '  <-- NEWER' if os.path.getmtime(f) > cur_mtime else ''
            print(f"    {mt}  {os.path.getsize(f):>12,}  {f}{newer}")

    # --- 1. qc_metadata.csv: the store's own index ---
    section("1. qc_metadata.csv (store index)")
    if not os.path.exists(QC):
        print(f"  MISSING: {QC}")
        return
    qc = pd.read_csv(QC, low_memory=False)
    print(f"  rows: {len(qc):,}")
    print(f"  cols: {list(qc.columns)}")
    idc = [c for c in qc.columns if any(k in c.lower() for k in ('file', 'id', 'path'))]
    print(f"\n  id/path cols: {idc}")
    if idc:
        print(qc[idc].head(10).to_string())
    qc_dates = show_dates(qc, 'qc_metadata')

    # --- 2. Does qc cover post-cutoff ECGs (i.e. more data than current key)? ---
    section("2. More data than the current key?")
    has_new = any(int((pd.to_datetime(qc[c], errors='coerce') > CUTOFF).sum()) > 0 for c in qc_dates)
    print(f"  qc_metadata has ECGs after {CUTOFF}: {has_new}")
    if os.path.exists(CUR_KEY):
        cur = pd.read_csv(CUR_KEY, low_memory=False, nrows=200_000)
        print(f"\n  current key sample ({len(cur):,} rows):")
        show_dates(cur, 'current_key')

    # --- 3. Resolve a few keys against disk ---
    section("3. Do the index keys resolve to .npy on disk?")
    # pick the most path-like column
    cand = [c for c in idc if qc[c].astype(str).str.contains('/').any()] or idc
    if not cand:
        print("  no id/path column found")
        return
    key_col = cand[0]
    print(f"  using key column: '{key_col}'")
    hits = miss = 0
    for v in qc[key_col].dropna().astype(str).head(20):
        stem = v[:-4] if v.lower().endswith(('.npy', '.dcm', '.xml')) else v
        p = os.path.join(STORE, stem + '.npy')
        ok = os.path.exists(p)
        hits += ok; miss += (not ok)
        if hits + miss <= 8:
            print(f"    {'OK ' if ok else 'MISS'}  {p}")
    print(f"  resolved {hits}/{hits+miss} of first 20")
    if hits:
        sample = os.path.join(STORE, str(qc[key_col].dropna().astype(str).iloc[0]).rstrip('.npy') + '.npy')
        try:
            arr = np.load(sample, allow_pickle=True)
            print(f"  sample shape={arr.shape} dtype={arr.dtype} "
                  f"min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f} (mV if |max|<50)")
        except Exception as e:
            print(f"  load sample failed: {e}")

    # --- 4. Cross-check against the cohort fileID, if present ---
    section("4. Cohort fileID vs store key")
    coh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabs', 'cohort_test.csv')
    if os.path.exists(coh_path):
        coh = pd.read_csv(coh_path, low_memory=False)
        fcols = [c for c in coh.columns if 'ile' in c.lower()]
        print(f"  cohort_test cols with 'ile': {fcols}")
        print(coh[fcols].head(8).to_string())
    else:
        print(f"  cohort_test.csv not found at {coh_path}")


if __name__ == '__main__':
    main()
