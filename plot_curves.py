"""
Plot train vs val AUROC/loss per epoch to spot overfitting (divergence point).

Usage:
    python plot_curves.py                       # latest *_epochs.csv in models/
    python plot_curves.py models/amyloid_<run_id>_epochs.csv
"""
import os
import sys
import glob

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import project_constants as project

MODEL_DIR = os.path.join(project.project_root, 'models')


def latest_epoch_csv():
    files = glob.glob(os.path.join(MODEL_DIR, '*_epochs.csv'))
    if not files:
        raise FileNotFoundError(f"No *_epochs.csv in {MODEL_DIR}")
    return max(files, key=os.path.getmtime)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_epoch_csv()
    print(f"Reading: {path}")
    df = pd.read_csv(path, sep=';')
    df = df.reset_index(drop=True)
    epochs = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUROC
    if 'auroc' in df and 'val_auroc' in df:
        axes[0].plot(epochs, df['auroc'], 'o-', label='train auroc')
        axes[0].plot(epochs, df['val_auroc'], 's-', label='val auroc')
        best = df['val_auroc'].idxmax()
        axes[0].axvline(best + 1, color='gray', ls='--', alpha=0.6,
                        label=f'best val (ep {best + 1}: {df["val_auroc"].max():.3f})')
        axes[0].set_title('AUROC — divergence = train climbs, val flattens')
        axes[0].set_xlabel('epoch'); axes[0].set_ylabel('AUROC'); axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Loss
    if 'loss' in df and 'val_loss' in df:
        axes[1].plot(epochs, df['loss'], 'o-', label='train loss')
        axes[1].plot(epochs, df['val_loss'], 's-', label='val loss')
        axes[1].set_title('Loss — overfitting = val_loss rises while train falls')
        axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss'); axes[1].legend()
        axes[1].grid(alpha=0.3)

    out = path.replace('.csv', '_curves.png')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved: {out}")

    # Text summary of divergence
    if 'auroc' in df and 'val_auroc' in df:
        gap = df['auroc'] - df['val_auroc']
        print("\nepoch | train_auroc | val_auroc | gap")
        for i in range(len(df)):
            flag = '  <- best val' if i == df['val_auroc'].idxmax() else ''
            print(f"  {i+1:>3} |   {df['auroc'][i]:.4f}   |  {df['val_auroc'][i]:.4f}  | "
                  f"{gap[i]:+.4f}{flag}")


if __name__ == '__main__':
    main()
