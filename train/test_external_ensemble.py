"""
Score the SEED-ENSEMBLE on the EXTERNAL image cohorts (WVU, Northwell, Greece,
SCAN-MP) and compare it, per cohort, to the single members + yesterday's-best +
production, with a paired bootstrap CI of the gain.

This reuses test_external_cohorts.load_external_images() VERBATIM, so the deploy
preprocessing (L->RGB->resize300 [0,1], optional CROP=1 YOLO / ADJUST=1 bright-
contrast) is byte-identical to the validated single-model external eval. Same env
knobs apply:  CROP=1 YOLO_WEIGHTS=... ADJUST=1 python train/test_external_ensemble.py

Models are HARDCODED via ensemble_eval (MEMBER_RUN_IDS, YDAY_MODEL, PROD_MODEL).
Ensemble = soft-average of the 3 seed members' probabilities (refs NOT averaged).

External cohorts are ~1 ECG/patient, so metrics are per-image (matches the existing
external eval + bootstrap_ci.py). Reports, per cohort:
  - per-member + yday + prod + ENSEMBLE AUROC/AUPRC
  - paired bootstrap Δ(ENSEMBLE - prod), Δ(ENSEMBLE - yday), Δ(ENSEMBLE - best member)
    with 95% CI. Δ-CI excluding 0 => the ensemble's edge is unlikely to be chance.

Saves: tabs/external_ensemble_preds.csv     (per-image, all pred_ cols incl pred_ENS)
       tabs/external_ensemble_metrics.csv   (per cohort x model)
"""
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import loss_fn
from eval import sensitivity_at_specificity, specificity_at_sensitivity
import test_external_cohorts as tec
from ensemble_eval import (MEMBER_RUN_IDS, _best_dir_for_run, YDAY_MODEL,
                           PROD_MODEL, _ci, N_BOOT, BOOT_SEED)


def main():
    # --- resolve models (hardcoded; same as internal ensemble_eval) ---
    members = []
    for label, run_id in MEMBER_RUN_IDS:
        ckpt = _best_dir_for_run(run_id)
        if ckpt is None:
            print(f"WARNING: no checkpoint for run {run_id}, skipping.")
            continue
        members.append((label, ckpt))
    assert len(members) >= 2, f"Need >=2 members, resolved {len(members)}."
    refs = [(lbl, p) for lbl, p in [('yday', YDAY_MODEL), ('prod', PROD_MODEL)]
            if os.path.isdir(p) or os.path.exists(p)]

    print("Ensemble members (soft-averaged):")
    for lbl, p in members:
        print(f"  [{lbl}] {p}")
    print("Reference models (scored, not averaged):")
    for lbl, p in refs:
        print(f"  [{lbl}] {p}")

    # --- load external images ONCE (identical render to single-model eval) ---
    df, X = tec.load_external_images()

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # --- predict each model once (one in VRAM at a time) ---
    member_cols, ref_cols = [], []
    for lbl, path in members:
        model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn})
        df[f'pred_{lbl}'] = tec.predict(model, X)
        member_cols.append(f'pred_{lbl}')
        del model
        tf.keras.backend.clear_session()
    df['pred_ENS'] = df[member_cols].mean(axis=1)
    for lbl, path in refs:
        model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn})
        df[f'pred_{lbl}'] = tec.predict(model, X)
        ref_cols.append(f'pred_{lbl}')
        del model
        tf.keras.backend.clear_session()

    all_cols = member_cols + ['pred_ENS'] + ref_cols
    rng = np.random.default_rng(BOOT_SEED)

    rows = []
    print("\n" + "=" * 78)
    print("EXTERNAL COHORTS — ENSEMBLE vs members/yday/prod (per-image)")
    print("=" * 78)
    for name, g in df.groupby('cohort'):
        y = g['label'].to_numpy()
        if len(np.unique(y)) < 2:
            print(f"\n{name}: only one class — skipped")
            continue
        n = len(g)
        print(f"\n=== {name}  (n={n}, pos={int(y.sum())}) ===")
        for c in all_cols:
            s = g[c].to_numpy()
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
            sens, _ = sensitivity_at_specificity(y, s, 0.90)
            spec, _ = specificity_at_sensitivity(y, s, 0.90)
            tag = c.replace('pred_', '')
            print(f"  {tag:8s} AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                  f"Sens@90spec={sens:.4f} Spec@90sens={spec:.4f}")
            rows.append({'cohort': name, 'model': tag, 'n': n, 'pos': int(y.sum()),
                         'auroc': round(auroc, 4), 'auprc': round(auprc, 4),
                         'sens_90spec': sens, 'spec_90sens': spec})

        # paired bootstrap: shared resample indices across models -> paired deltas
        idx_sets = []
        guard = 0
        while len(idx_sets) < N_BOOT and guard < N_BOOT * 50:
            idx = rng.integers(0, n, n)
            guard += 1
            if len(np.unique(y[idx])) == 2:
                idx_sets.append(idx)
        if len(idx_sets) < N_BOOT:
            print(f"  (only {len(idx_sets)} valid boots — small/imbalanced cohort)")
        if not idx_sets:
            continue

        def boot(col):
            s = g[col].to_numpy()
            return np.array([roc_auc_score(y[i], s[i]) for i in idx_sets])

        b_ens = boot('pred_ENS')
        member_point = {c: roc_auc_score(y, g[c].to_numpy()) for c in member_cols}
        best_member = max(member_point, key=member_point.get)
        compare = [('best member ' + best_member.replace('pred_', ''), best_member)] + \
                  [(c.replace('pred_', '') + ' (ref)', c) for c in ref_cols]
        elo, ehi = _ci(b_ens)
        print(f"  ENSEMBLE AUROC={roc_auc_score(y, g['pred_ENS'].to_numpy()):.4f} "
              f"[{elo:.4f},{ehi:.4f}]  ({len(idx_sets)} boots)")
        for cname2, col in compare:
            diff = b_ens - boot(col)
            dlo, dhi = _ci(diff)
            verdict = ("ENS wins (CI>0)" if dlo > 0 else
                       f"{cname2} wins (CI<0)" if dhi < 0 else "overlaps 0 (n.s.)")
            print(f"    Δ(ENS - {cname2}) = {diff.mean():+.4f} [{dlo:+.4f},{dhi:+.4f}]  -> {verdict}")

    out = pd.DataFrame(rows)
    metrics_path = os.path.join(project.tabs_path, 'external_ensemble_metrics.csv')
    out.to_csv(metrics_path, index=False)
    keep = ['cohort', 'label', 'path'] + all_cols
    preds_path = os.path.join(project.tabs_path, 'external_ensemble_preds.csv')
    df[[c for c in keep if c in df.columns]].to_csv(preds_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved preds:   {preds_path}")


if __name__ == '__main__':
    main()
