"""
Bootstrap 95% CIs for the external-cohort AUROC/AUPRC, per model, from the
per-sample predictions saved by test_external_cohorts.py (tabs/external_preds.csv).

Compare NEW vs OLD by CI overlap (no DeLong). Also reports a paired bootstrap CI of
the difference (NEW - OLD) as a bonus — if that CI excludes 0, the gap is unlikely
to be chance. Resampling is at the row (image) level; the external folders are ~1
ECG/patient, so this approximates patient-level. If you later add a patient_id, switch
RESAMPLE_UNIT to group by it.

Run on the box:  python train/bootstrap_ci.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

PREDS = os.path.join(project.tabs_path, 'external_preds.csv')
N_BOOT = 2000
SEED = 20250910
ALPHA = 0.05


def _ci(vals):
    lo, hi = np.percentile(vals, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return lo, hi


def main():
    if not os.path.exists(PREDS):
        print(f"Missing {PREDS} — run test_external_cohorts.py first.")
        return
    df = pd.read_csv(PREDS)
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    tags = [c.replace('pred_', '') for c in pred_cols]
    rng = np.random.default_rng(SEED)

    summary = []
    for cohort, g in df.groupby('cohort'):
        y = g['label'].to_numpy()
        if len(np.unique(y)) < 2:
            print(f"{cohort}: single class, skip")
            continue
        n = len(g)
        scores = {t: g[f'pred_{t}'].to_numpy() for t in tags}

        # pre-draw bootstrap index sets (shared across models -> paired)
        idx_sets = []
        while len(idx_sets) < N_BOOT:
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) == 2:      # need both classes
                idx_sets.append(idx)

        print(f"\n=== {cohort}  (n={n}, pos={int(y.sum())}, {N_BOOT} boots) ===")
        boot_auroc = {}
        for t in tags:
            s = scores[t]
            pt_auroc = roc_auc_score(y, s)
            pt_auprc = average_precision_score(y, s)
            b_auroc = np.array([roc_auc_score(y[i], s[i]) for i in idx_sets])
            b_auprc = np.array([average_precision_score(y[i], s[i]) for i in idx_sets])
            boot_auroc[t] = b_auroc
            lo, hi = _ci(b_auroc)
            plo, phi = _ci(b_auprc)
            print(f"  {t:4s} AUROC={pt_auroc:.3f} [{lo:.3f}, {hi:.3f}]   "
                  f"AUPRC={pt_auprc:.3f} [{plo:.3f}, {phi:.3f}]")
            summary.append({'cohort': cohort, 'model': t, 'n': n, 'pos': int(y.sum()),
                            'auroc': round(pt_auroc, 4), 'auroc_lo': round(lo, 4), 'auroc_hi': round(hi, 4),
                            'auprc': round(pt_auprc, 4), 'auprc_lo': round(plo, 4), 'auprc_hi': round(phi, 4)})

        # paired difference NEW - OLD (same resample indices)
        if 'NEW' in boot_auroc and 'OLD' in boot_auroc:
            diff = boot_auroc['NEW'] - boot_auroc['OLD']
            dlo, dhi = _ci(diff)
            p_gt = float(np.mean(diff > 0))
            verdict = "NEW>OLD (CI excludes 0)" if dlo > 0 else \
                      "OLD>NEW (CI excludes 0)" if dhi < 0 else "overlaps 0 (n.s.)"
            print(f"  Δ AUROC (NEW-OLD) = {diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]  "
                  f"P(NEW>OLD)={p_gt:.2f}  -> {verdict}")

    out = pd.DataFrame(summary)
    p = os.path.join(project.tabs_path, 'external_bootstrap_ci.csv')
    out.to_csv(p, index=False)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
