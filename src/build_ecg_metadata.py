"""
Regenerate the flagged ECG-metadata key from the 4 source parquets.

This is lsd26's builder, repo-ized: same 4 sources, same fileID construction, but
paths come from project_constants (nfs root) and the output is a fresh dated CSV in
<signals>/data_LD/ so project_constants._latest_ecg_metadata() auto-picks it.

Why re-run: the pdcfs1 (Philips XML feed from YNHH) source parquet was refreshed
past the old July-2025 ceiling (now through ~present). Re-running pulls those new
ECGs into the key that build_cohort consumes.

Source -> fileID scheme (MUST match the on-disk layout, do not "fix"):
  metadata.parquet               2000-mid2021   fileID = FileID minus .dcm
  ekg_2021_metadata.parquet      2021 H2        fileID = ekg_waveform_2021/<id>
  pdcfs1/ecg_metadata_pdcfs1     2022->present  fileID = pdcfs1/<yr>/<full[6:8]>/<fileBase>
  VNAXDS1VP/..._2022             2022           fileID = VNAXDS1VP/2022/<mm>/<id>

Run on the box:  python src/build_ecg_metadata.py
"""
import os
import sys
import glob
import datetime as dt
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

SIG  = project.signals            # /mnt/nfs_yale_ecg_signals  (raw signals + sources)
RAID = project.signals_base       # /mnt/raid0/bb2238/signals  (QC store + legacy)
OUT  = os.path.join(SIG, 'data_LD',
                    f"ecg_metadata_flagged_01jan2000_to_{dt.date.today():%d%b%Y}.csv".lower())

RENAME = {'StudyDate': 'ECGDate', 'PatientBirthDate': 'PatientBirthDate_ECGData',
          'PatientSex': 'PatientSex_ECGData', 'EthnicGroup': 'EthnicGroup_ECGData',
          'PatientAge': 'PatientAge_ECGData'}


def _load_original():
    d = pd.read_parquet(os.path.join(SIG, 'data_LD/ecg-echo-raw-data/metadata.parquet'))
    d['MRN'] = d['PatientID']
    d = d.rename(columns=RENAME)
    d['ECGDate'] = pd.to_datetime(d['ECGDate'], errors='coerce')
    d['which_metadata'] = '2000_to_mid2021'
    d['fileID'] = d['FileID'].str.replace('.dcm', '', regex=False)
    return d


def _load_2021h2():
    d = pd.read_parquet(os.path.join(SIG, 'data_LD/ecg-echo-raw-data/ekg_2021_metadata.parquet'))
    d['PatientID'] = d['PatientID'].str.replace('MRMR', 'MR', regex=False)
    d['MRN'] = d['PatientID']
    d = d.rename(columns=RENAME)
    d['ECGDate'] = pd.to_datetime(d['ECGDate'], errors='coerce')
    d['which_metadata'] = 'second_half_of_2021'
    d['fileID'] = ('ekg_waveform_2021/' +
                   d['FileID'].str.replace(r"ekg_2021/\d{4}/\d{2}/\d{2}/", '', regex=True)
                              .str.replace('.xml', '', regex=False))
    return d


def _load_pdcfs1():
    d = pd.read_parquet(os.path.join(SIG, 'numpy/pdcfs1/ecg_metadata_pdcfs1.parquet'))
    d['MRN'] = d['PatientID']
    d = d.rename(columns=RENAME)
    d['ECGDate'] = pd.to_datetime(d['ECGDate'], errors='coerce')
    d['which_metadata'] = 'pdcfs1'

    def build_fid(fb):
        fb = str(fb)
        full = fb[:8]          # YYYYMMDD
        return f"pdcfs1/{full[:4]}/{full[6:]}/{fb}"   # lsd26's exact scheme (full[6:] = DD)
    d['fileID'] = d['fileBase'].map(build_fid)
    return d


def _load_vnax():
    d = pd.read_parquet(os.path.join(SIG, 'numpy/VNAXDS1VP/ecg_metadata_VNAXDS1VP_2022.parquet'))
    d['MRN'] = d['PatientID']
    # fileID built from StudyDate BEFORE to_datetime (needs the raw string form)
    d['fileID'] = 'VNAXDS1VP/2022/' + d['StudyDate'].str[4:6] + '/' + d['FileID'].str[9:-4]
    d = d.rename(columns=RENAME)
    d['ECGDate'] = pd.to_datetime(d['ECGDate'], errors='coerce')
    d['which_metadata'] = 'VNAXDS1VP'
    return d


# signal-existence check (fast; os.path.exists, not np.load). Searches the same
# places make_plot does: raid store, raid legacy, nfs raw.
_SEARCH = [
    (RAID, 'preprocessed/all_ecgs'),
    (SIG,  'numpy_rp'), (SIG, 'numpy'),
    (RAID, 'numpy_rp'), (RAID, 'numpy'),
]


def _exists(fid):
    for base, sub in _SEARCH:
        if os.path.exists(os.path.join(base, sub, fid + '.npy')):
            return True
    return False


def main():
    print(f"Output -> {OUT}")
    parts = [_load_original(), _load_2021h2(), _load_pdcfs1(), _load_vnax()]
    for p, name in zip(parts, ['original', '2021h2', 'pdcfs1', 'VNAXDS1VP']):
        mx = pd.to_datetime(p['ECGDate'], errors='coerce').max()
        print(f"  {name:10s} rows={len(p):>9,}  max ECGDate={mx}")

    full = pd.concat(parts, ignore_index=True, sort=False)
    full = full.drop_duplicates(subset=['fileID'])
    print(f"concat + dedup: {len(full):,} rows")

    # pre-procedure ECG flag (optional; build_cohort doesn't require it, kept for parity)
    proc_path = os.path.join(SIG, 'data_LD/data_jan2024/cv_procs/'
                                  'earliest_cv_procs_for_all_jdat_cohorts_26jan2024.csv')
    if os.path.exists(proc_path):
        proc = pd.read_csv(proc_path)
        proc['EARLIEST_PROC_DATE'] = pd.to_datetime(proc['EARLIEST_PROC_DATE'], errors='coerce')
        full = full.merge(proc, on='MRN', how='left')
        full['ecg_before_proc'] = full['ECGDate'] < full['EARLIEST_PROC_DATE'].fillna(pd.Timestamp.now())
    else:
        print(f"  (skipped proc flag — {proc_path} not found)")

    full = full[full['MRN'].notna()]
    print(f"after MRN-notna: {len(full):,}")

    # fails_to_load: reuse the previous flagged file as a known-good cache so only the
    # NEW fileIDs hit the disk. Cuts a 5M-row scan down to the refreshed rows.
    prev = project.ecg_metadata_file
    known = set()
    if os.path.exists(prev) and os.path.abspath(prev) != os.path.abspath(OUT):
        known = set(pd.read_csv(prev, usecols=['fileID'])['fileID'].astype(str))
        print(f"known-good from {os.path.basename(prev)}: {len(known):,}")
    fids = full['fileID'].astype(str)
    to_check = sorted(set(fids) - known)
    print(f"new fileIDs to disk-check: {len(to_check):,}  (workers={cpu_count()})")
    ok = {}
    if to_check:
        with Pool(cpu_count()) as pool:
            for f, res in zip(to_check, pool.map(_exists, to_check, chunksize=512)):
                ok[f] = res
    loads = fids.map(lambda f: True if f in known else ok.get(f, False))
    print(f"loadable: {int(loads.sum()):,} / {len(full):,}  (dropping {int((~loads).sum()):,})")
    full = full[loads.values]

    full.to_csv(OUT, index=False)
    print(f"\nWrote {len(full):,} rows -> {OUT}")
    print("project_constants._latest_ecg_metadata() will now auto-pick this file.")
    print("Next: python src/build_cohort.py  (then matching.R / train).")


if __name__ == '__main__':
    main()
