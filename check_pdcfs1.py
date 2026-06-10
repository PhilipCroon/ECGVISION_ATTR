"""
Settle the two pdcfs1 assumptions BEFORE regenerating metadata or retraining.
Both are cheap (qc_metadata + a couple of files); a wrong assumption silently
corrupts the new training set. Run on the box:  python check_pdcfs1.py

Blocker 1 — path scheme: does build_ecg_metadata.build_fid produce the SAME
  relative path the store actually uses (qc_metadata.fid_rel)? lsd26's scheme uses
  full_date[6:8] = DAY, not month; only correct if the store was built that way.

Blocker 2 — scale/shape: is reading the RAW nfs file with utils.make_plot._load_norm
  identical to the consolidated store array? qc_metadata's scaled_by / has_padding /
  original_shape columns suggest the raw->store transform may NOT be trivial.

Both PASS  -> safe to regenerate + retrain.
Either FAIL -> fix before spending hours (or ask bb2238 to consolidate to the store).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import project_constants as project

STORE = os.path.join(project.signals_base, 'preprocessed', 'all_ecgs')
NFS   = project.signals
QC    = os.path.join(STORE, 'qc_metadata.csv')


# --- mirror of build_ecg_metadata._load_pdcfs1.build_fid (KEEP IN SYNC) ---
def build_fid(fb):
    fb = str(fb)
    full = fb[:8]
    return f"pdcfs1/{full[:4]}/{full[6:]}/{fb}"


# --- mirror of utils.make_plot loader (KEEP IN SYNC) ---
def load_store(path):
    # raid QC store: already mV, orient only.
    raw = np.asarray(np.load(path, allow_pickle=True))
    if raw.shape[0] == 12 and raw.shape[1] >= 5000:
        raw = raw.T
    return raw[0:5000, :]


def load_raw_pdcfs1(path):
    # raw nfs numpy/ for a pdcfs1 fid ('p' prefix): transpose, /200, take 5000.
    signal = np.array(np.load(path, allow_pickle=True))
    signal = signal.T
    return signal[0:5000, :] / 200


def main():
    qc = pd.read_csv(QC, low_memory=False)
    pdc = qc[qc['fid_rel'].astype(str).str.startswith('pdcfs1/')].copy()
    print(f"pdcfs1 rows in store index (qc_metadata): {len(pdc):,}")
    if len(pdc) == 0:
        print("\n!! No pdcfs1 in the store yet. The new batch is nfs-only; Blocker 2 can't be\n"
              "   verified against the store. Strongly consider asking bb2238 to consolidate.")
        return

    print("\nsample fid_rel:")
    print(pdc['fid_rel'].head(8).to_string())
    seg = pdc['fid_rel'].str.split('/').str[2]
    print("\n3rd path segment distribution (if values run 01..12 -> month; up to 31 -> day):")
    print(sorted(seg.dropna().unique().tolist())[:35])

    # ---- Blocker 1 ----
    print("\n" + "=" * 60 + "\nBLOCKER 1: build_fid vs store fid_rel\n" + "=" * 60)
    pdc['base'] = pdc['fid_rel'].str.split('/').str[-1]
    pdc['built'] = pdc['base'].map(build_fid)
    match = float((pdc['built'] == pdc['fid_rel']).mean())
    print(f"  build_fid(base) == fid_rel : {match:.2%} of {len(pdc):,}")
    if match < 1.0:
        bad = pdc[pdc['built'] != pdc['fid_rel']]
        print("  MISMATCH examples (built -> actual):")
        print(bad[['built', 'fid_rel']].head(8).to_string(index=False))
        print("  >>> FAIL: regenerated pdcfs1 keys would not resolve. Fix build_fid.")
    else:
        print("  >>> PASS: build_fid matches the store layout.")

    # ---- Blocker 2 ----
    print("\n" + "=" * 60 + "\nBLOCKER 2: _load_norm(raw nfs) vs store array\n" + "=" * 60)
    # /200 is confirmed (scaled_by=200); the open question is the 5500->5000 step.
    # Try candidate transforms on raw (12,5500)/200 and report which matches the store.
    import scipy.signal as _ss

    def candidates(raw):
        # raw: np.load of nfs file. Orient to (N,12), scale /200, then reduce N->5000.
        a = np.asarray(raw)
        if a.shape[0] == 12:
            a = a.T                      # (5500,12)
        a = a / 200.0
        n = a.shape[0]
        out = {}
        out['first[0:5000]']      = a[0:5000, :]
        out['last[-5000:]']       = a[n-5000:, :]
        out['center']             = a[(n-5000)//2:(n-5000)//2+5000, :]
        out['resample5000']       = _ss.resample(a, 5000, axis=0)
        out['decimate(stride)']   = a[::max(1, n//5000), :][:5000, :]
        return out

    checked = 0
    agg = {}
    for fr in pdc['fid_rel']:
        sp = os.path.join(STORE, fr + '.npy')
        rp = os.path.join(NFS, 'numpy', fr + '.npy')
        if os.path.exists(sp) and os.path.exists(rp):
            s = load_store(sp)
            raw = np.load(rp, allow_pickle=True)
            print(f"\n  {fr}  store={s.shape}  raw={np.asarray(raw).shape}")
            for name, cand in candidates(raw).items():
                if cand.shape != s.shape:
                    print(f"    {name:18s} shape={cand.shape}  (mismatch)")
                    continue
                d = float(np.abs(s - cand).max())
                agg.setdefault(name, []).append(d)
                print(f"    {name:18s} max|store-cand|={d:.6f}")
            checked += 1
            if checked >= 5:
                break
    if checked == 0:
        print("  No pdcfs1 fileID found in BOTH store and nfs numpy/ — cannot compare directly.")
    else:
        print("\n  --- transform ranking (mean worst-diff over samples) ---")
        ranked = sorted(((np.mean(v), k) for k, v in agg.items()))
        for m, k in ranked:
            print(f"    {k:18s} mean max|diff|={m:.6f}")
        best_m, best_k = ranked[0]
        print(f"\n  raw-signal BEST: {best_k}  (mean {best_m:.6f}) — likely baseline-wander residual")

    # DECISIVE: make_plot removes baseline via median_filter(500). The model sees the
    # POST-filter signal, not the raw. Compare store vs raw[0:5000]/200 AFTER that filter.
    print("\n" + "=" * 60 + "\nBLOCKER 2b: AFTER make_plot baseline removal (what the model sees)\n" + "=" * 60)
    from scipy.ndimage import median_filter as _mf

    def mp(sig):  # mirror make_plot: proc - median_filter(proc, (500,1))
        return sig - _mf(sig, size=(500, 1))

    checked = 0
    worst = 0.0
    for fr in pdc['fid_rel']:
        sp = os.path.join(STORE, fr + '.npy')
        rp = os.path.join(NFS, 'numpy', fr + '.npy')
        if os.path.exists(sp) and os.path.exists(rp):
            s = load_store(sp)
            r = load_raw_pdcfs1(rp)          # raw.T[0:5000]/200
            if s.shape == r.shape:
                d = float(np.abs(mp(s) - mp(r)).max())
                worst = max(worst, d)
                print(f"  {fr}  post-filter max|diff|={d:.6f}")
            checked += 1
            if checked >= 5:
                break
    if checked:
        print(f"\n  worst post-filter max|diff| = {worst:.6f}")
        print("  >>> PASS — images match after baseline removal; /200 + [0:5000] is safe."
              if worst < 1e-3 else
              "  >>> FAIL — diff survives median filter; not just baseline. Need bb2238 to consolidate.")

    print("\nqc_metadata transform fields (why raw->store may be non-trivial):")
    cols = [c for c in ['fid_rel', 'original_shape', 'scaled_by', 'has_padding',
                        'unpadded_length', 'final_shape', 'has_nan', 'status'] if c in pdc.columns]
    print(pdc[cols].head(6).to_string(index=False))
    if 'scaled_by' in pdc.columns:
        print("\n  scaled_by value counts:", pdc['scaled_by'].value_counts().head().to_dict())
    if 'has_padding' in pdc.columns:
        print("  has_padding value counts:", pdc['has_padding'].value_counts().to_dict())


if __name__ == '__main__':
    main()
