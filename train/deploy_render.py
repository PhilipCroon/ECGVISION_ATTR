"""
Vendored deployment ECG render — copied VERBATIM from ynhh-apis
amyloid-api/ecg/ecg.py so the test render matches production without cloning the
API repo. Source functions: butter_lowpass, butter_lowpass_filter,
custom_ecg_plot, process_ecg_plot_from_signal.

Keep in sync with ecg.ecg if the deployment render changes. Do not "improve" —
fidelity to the deployed pipeline is the whole point.
"""
import os
import gc
from math import ceil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import butter, lfilter
from scipy.ndimage import median_filter


def butter_lowpass(highcut, sampfreq, order):
    """Supporting function. Prepare data and call the scipy butter function."""
    nyquist_freq = .5 * sampfreq
    high = highcut / nyquist_freq
    num, denom = butter(order, high, btype='lowpass')
    return num, denom


def butter_lowpass_filter(data, highcut, sampfreq, order):
    """Apply the Butterworth lowpass filter to the waveform."""
    num, denom = butter_lowpass(highcut, sampfreq, order=order)
    return lfilter(num, denom, data)


def custom_ecg_plot(
        ecg,
        sample_rate=500,
        title='ECG 12',
        lead_index=None,
        lead_order=None,
        style=None,
        columns=2,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
        debug=False):
    """Plot multi-lead ECG chart."""

    lead_index = ['I', 'II', 'III', 'I', 'aVR', 'aVL', 'aVF', '', 'V1', 'V2', 'V3', '', 'V4', 'V5', 'V6', '']

    assert isinstance(ecg, np.ndarray), "ECG input must be a NumPy array"
    assert ecg.ndim == 2, f"ECG must be 2D (leads x samples), got shape {ecg.shape}"
    m, n = ecg.shape
    assert m >= 1 and n >= 100, f"ECG must have at least 1 lead and 100 samples, got shape {ecg.shape}"

    if lead_index is None:
        lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'][:m]
    assert len(lead_index) == m, f"lead_index must have {m} names, but got {len(lead_index)}"

    if lead_order is None:
        lead_order = list(range(m))
    assert max(lead_order) < m, "lead_order contains invalid lead index"

    secs = n / sample_rate
    leads = len(lead_order)
    rows = int(ceil(leads / columns))
    display_factor = 1
    line_width = 0.5

    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    color_schemes = {
        'bw':       ((0.4, 0.4, 0.4), (0.75, 0.75, 0.75), (0, 0, 0)),
        'bw_alt':   ((.6, .6, .6),    (0.9, 0.9, 0.9),     (0, 0, 0)),
        'black_pink': ((.65, .65, .65), (1, 0.7, 0.7),     (0, 0, 0)),
        'blue_pink': ((1, 0, 0),      (1, 0.7, 0.7),       (0, 0, 0.7)),
        None:       ((1, 0, 0),       (1, 0.7, 0.7),       (0, 0, 0.7))
    }
    color_major, color_minor, color_line = color_schemes.get(style, color_schemes[None])

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, 0.2))
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for c in range(columns):
        for i in range(rows):
            lead_idx = c * rows + i
            if lead_idx >= leads:
                continue

            t_lead = lead_order[lead_idx]
            y_offset = -(row_height / 2) * ceil(i % rows)
            x_offset = secs * c if c > 0 else 0

            if debug:
                print(f"Plotting lead {t_lead}: {lead_index[t_lead]}")

            if show_separate_line and c > 0:
                try:
                    vline_y = ecg[t_lead][0] + y_offset
                    ax.plot([x_offset, x_offset], [vline_y - 0.3, vline_y + 0.3], linewidth=line_width * display_factor, color=color_line)
                except Exception as e:
                    print(f"Error drawing separator for lead {t_lead}: {e}")

            if show_lead_name:
                ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)

            try:
                x_vals = np.linspace(0, n / sample_rate, num=n, endpoint=False) + x_offset
                y_vals = ecg[t_lead] + y_offset

                if np.all(y_vals == 0):
                    print(f"  Lead {t_lead} ('{lead_index[t_lead]}') is flat (all zeros)")

                ax.plot(x_vals, y_vals, linewidth=line_width * display_factor, color=color_line)

            except Exception as e:
                print(f"Error plotting lead {t_lead} ('{lead_index[t_lead]}'): {e}")

    return fig


def process_ecg_plot_from_signal(signal, fid: str, save_path: str):
    """Plot a 12-lead ECG (12,5000) and save as PNG. signal expected in mV."""
    if signal.shape != (12, 5000):
        raise ValueError(f"Invalid ECG shape from {fid}: expected (12, 5000), got {signal.shape}")

    proc_signal = signal / 1000
    proc_signal2 = proc_signal - median_filter(proc_signal, size=(500, 1))  # computed, unused (as in source)
    full = signal.T  # (5000, 12)

    col1 = full[0:1250, 0:3].T
    col1a = full[0:1250, 0:1].T
    col2 = full[1250:2500, 3:6].T
    col2a = full[1250:2500, 0:1].T
    col3 = full[2500:3750, 6:9].T
    col3a = full[2500:3750, 0:1].T
    col4 = full[3750:5000, 9:12].T
    col4a = full[3750:5000, 0:1].T

    newplot1a = np.vstack((col1, col1a, col2, col2a, col3, col3a, col4, col4a))

    fig = custom_ecg_plot(
        newplot1a,
        sample_rate=500,
        title="",
        show_separate_line=True,
        columns=4,
        lead_index=['I', 'II', 'III', 'I', 'aVR', 'aVL', 'aVF', '', 'V1', 'V2', 'V3', '', 'V4', 'V5', 'V6', ''],
        style='bw_alt'
    )

    ax = fig.axes[0]
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                   right=False, labelleft=False, labelbottom=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
