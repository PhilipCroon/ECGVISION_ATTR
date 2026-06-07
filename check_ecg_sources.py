"""
Feasibility check for ingesting ECGs after the current July-2025 ceiling.

The flagged metadata key is built (by lsd26's script) from 4 source parquets. The
newest ECG you can possibly use = the max ECGDate across those sources. pdcfs1 is
the only rolling one. This script reports each source's date range + post-cutoff
count, and globs for any NEWER source parquet that isn't wired in yet.

Run on the box:  python check_ecg_sources.py
Read-only.
"""
import os
import glob
import pandas as pd

CUTOFF = '2025-07-17'

# Mount root differs by alias; try both. lsd26 script used /mnt/yale-ecg-signals,
# project_constants uses /mnt/nfs_yale_ecg_signals.
ROOTS = ['/mnt/yale-ecg-signals', '/mnt/nfs_yale_ecg_signals']
RAID  = '/mnt/raid0/bb2238/signals'

SOURCES = [
    ('original 2000-mid2021', 'data_LD/ecg-echo-raw-data/metadata.parquet'),
    ('ekg_2021 H2',           'data_LD/ecg-echo-raw-data/ekg_2021_metadata.parquet'),
    ('pdcfs1 (rolling)',      'numpy/pdcfs1/ecg_metadata_pdcfs1.parquet'),
    ('VNAXDS1VP 2022',        'numpy/VNAXDS1VP/ecg_metadata_VNAXDS1VP_2022.parquet'),
]


def find(rel):
    for r in ROOTS + [RAID]:
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    return None


def report(label, path):
    print(f"\n--- {label} ---\n  {path}")
    if not path:
        print("  NOT FOUND under any root")
        return None
    try:
        d = pd.read_parquet(path)
    except Exception as e:
        print(f"  read failed: {e}")
        return None
    mt = pd.to_datetime(os.path.getmtime(path), unit='s')
    print(f"  file mtime: {mt}   rows: {len(d):,}")
    dc = [c for c in d.columns if 'date' in c.lower()] or \
         [c for c in d.columns if c in ('StudyDate',)]
    for c in dc[:4]:
        s = pd.to_datetime(d[c], errors='coerce')
        print(f"  {c:18s} min={s.min()} max={s.max()} after_{CUTOFF}={int((s>CUTOFF).sum()):,}")
    return d


def main():
    print("=" * 72 + "\nECG source-parquet coverage (decides the post-July-2025 ceiling)\n" + "=" * 72)
    found_roots = [r for r in ROOTS + [RAID] if os.path.isdir(r)]
    print(f"existing roots: {found_roots}")

    for label, rel in SOURCES:
        report(label, find(rel))

    # Any NEWER / extra metadata parquet not wired into the builder?
    print("\n" + "=" * 72 + "\nOther *metadata*.parquet under the signal trees (possible new sources)\n" + "=" * 72)
    seen = set()
    for r in found_roots:
        for p in glob.glob(os.path.join(r, '**', '*metadata*.parquet'), recursive=True):
            if p in seen:
                continue
            seen.add(p)
            mt = pd.to_datetime(os.path.getmtime(p), unit='s')
            print(f"  {mt}  {os.path.getsize(p):>14,}  {p}")

    print("\nDecision:")
    print("  - if any source max ECGDate > 2025-07-17  -> post-cutoff ECGs EXIST;")
    print("    regenerate the flagged metadata (re-run the builder) to capture them.")
    print("  - else -> July 2025 is the hard ceiling; proceed with the current cohort.")


if __name__ == '__main__':
    main()
