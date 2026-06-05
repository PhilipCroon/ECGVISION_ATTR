# %%
import os
import sys
import math
import random
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy
import scipy.stats as stats
import scipy.signal
from scipy.ndimage import median_filter
from scipy import ndimage
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from PIL import Image, ImageFile, ImageEnhance
from skimage.transform import resize, rotate
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve, auc,
                             average_precision_score, precision_recall_curve)
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- Resolve signal base path from project_constants if available ---
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
try:
    import project_constants as _pc
    _SIGNALS_BASE = _pc.signals_base
except (ImportError, AttributeError):
    _SIGNALS_BASE = '/mnt/raid0/bb2238/signals'


# %%

def fix_brightness_and_contrast(img):
    PIL_Image = Image.fromarray(img)
    PIL_Imagetemp = PIL_Image.convert('L')
    temp_bright = np.mean(resize(np.array(PIL_Imagetemp),(300,300))[50:250,50:250])
    counter = 0
    while((temp_bright >.89) or (temp_bright <.87)):
        if counter > 100:
            break
        if (temp_bright < .87):
            enhancer = ImageEnhance.Brightness(PIL_Imagetemp)
            PIL_Imagetemp = enhancer.enhance(1.05)
            temp_bright = np.mean(resize(np.array(PIL_Imagetemp),(300,300))[50:250,50:250])
            counter = counter + 1
        else:
            enhancer = ImageEnhance.Brightness(PIL_Imagetemp)
            PIL_Imagetemp = enhancer.enhance(0.95)
            temp_bright = np.mean(resize(np.array(PIL_Imagetemp),(300,300))[50:250,50:250])
            counter = counter + 1

    temp_con = np.std(resize(np.array(PIL_Imagetemp),(300,300))[50:250,50:250])

    counter = 0
    while((temp_con >.11) or (temp_con <.09)):
        if counter > 100:
            break

        if (temp_con < 0.09):
            enhancer = ImageEnhance.Contrast(PIL_Imagetemp)
            PIL_Imagetemp = enhancer.enhance(1.05)
            temp_con = np.std(resize(np.array(PIL_Imagetemp),(300,300))[50:250,50:250])
            counter = counter + 1

        else:
            enhancer = ImageEnhance.Contrast(PIL_Imagetemp)
            PIL_Imagetemp = enhancer.enhance(0.95)
            temp_con = np.std(resize(np.array(PIL_Imagetemp),(300,300))[50:250,50:250])
            counter = counter + 1

    return np.array(PIL_Imagetemp.convert('RGB'))


### Code to load data into the model for Training
class DataSequenceTrain(tf.keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, batch_size, mode):
        self.df = df
        self.bsz = batch_size
        self.mode = mode

        self.labels = self.df[class_names].values
        if self.mode == 'image':
            self.im_list = self.df['image_path'].tolist()
        else:
            self.im_list1 = self.df['fileID'].tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def get_batch_labels(self, idx):
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        if self.mode == 'image':
            return np.array([rotate(resize(np.array(Image.open(im).convert('L').convert('RGB')), (300,300)), np.random.uniform(-10,10,1)[0]) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])
        else:
            sigs = np.array([(np.load(os.path.join(_SIGNALS_BASE, 'numpy_rp', im+'.npy'))[0:5000,0:12] / 1000 - median_filter(np.load(os.path.join(_SIGNALS_BASE, 'numpy_rp', im+'.npy'))[0:5000,0:12] / 1000,size=(500,1))).T for im in self.im_list1[idx * self.bsz: (1 + idx) * self.bsz]])
            return sigs

    def __getitem__(self, idx):
        if self.mode == 'signal':
            sigs = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return sigs, batch_y
        if self.mode=='image':
            batch_x = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_x, batch_y


### Code to load data into the model for Testing
class DataSequenceTest(tf.keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, batch_size, mode, class_names):
        self.df = df
        self.bsz = batch_size
        self.mode = mode

        self.labels = self.df[class_names].values
        if self.mode == 'image':
            self.im_list = self.df['image_path'].tolist()
        else:
            self.im_list1 = self.df['fileID'].tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def get_batch_labels(self, idx):
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        if self.mode == 'image':
            return np.array([resize(np.array(Image.open(im).convert('L').convert('RGB')), (300,300)) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])
        else:
            sigs = np.array([(np.array(pd.read_parquet(os.path.join(_SIGNALS_BASE, 'waveform', im+'.parquet'), engine='fastparquet'))[0:5000,:] / 1000 - median_filter(np.array(pd.read_parquet(os.path.join(_SIGNALS_BASE, 'waveform', im+'.parquet'), engine='fastparquet'))[0:5000,:] / 1000,size=(500,1))).T for im in self.im_list1[idx * self.bsz: (1 + idx) * self.bsz]])
            return sigs

    def __getitem__(self, idx):
        if self.mode == 'signal':
            sigs = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return sigs, batch_y
        if self.mode=='image':
            batch_x = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_x, batch_y


class DataSequenceTest_Adjust(tf.keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, batch_size, mode):
        self.df = df
        self.bsz = batch_size
        self.mode = mode

        class_names = ['Under40']

        self.labels = self.df[class_names].values
        if self.mode == 'image':
            self.im_list = self.df['image_path'].tolist()
        else:
            self.im_list1 = self.df['fileID'].tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def get_batch_labels(self, idx):
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        if self.mode == 'image':
            return np.array([resize(fix_brightness_and_contrast(np.array(Image.open(im))), (300,300)) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])
        else:
            sigs = np.array([(np.array(pd.read_parquet(os.path.join(_SIGNALS_BASE, 'waveform', im+'.parquet'), engine='fastparquet'))[0:5000,:] / 1000 - median_filter(np.array(pd.read_parquet(os.path.join(_SIGNALS_BASE, 'waveform', im+'.parquet'), engine='fastparquet'))[0:5000,:] / 1000,size=(500,1))).T for im in self.im_list1[idx * self.bsz: (1 + idx) * self.bsz]])
            return sigs

    def __getitem__(self, idx):
        if self.mode == 'signal':
            sigs = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return sigs, batch_y
        if self.mode=='image':
            batch_x = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_x, batch_y


##### Stats Metrics Helpers
def TP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))

def TN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))

def FP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))

def FN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))

def accuracy(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    return (tp+tn)/(tp+tn+fp+fn)

def precision(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    return tp/(tp+fp)

def recall(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fn = FN(y,pred,th)
    return tp/(tp+fn)

def sensitivity(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fn = FN(y,pred,th)
    return tp/(tp+fn)

def specificity(y, pred, th=0.5):
    tn = TN(y,pred,th)
    fp = FP(y,pred,th)
    return tn/(tn+fp)

def fscore(y,pred,beta=1.0):
    p = precision(y,pred)
    r = recall(y,pred)
    return (1+beta**2)*p*r/((1+beta**2)*p+r)

def f1(y,pred,beta=1.0):
    p = precision(y,pred)
    r = recall(y,pred)
    return (1+beta**2)*p*r/((1+beta**2)*p+r)

def prcurve(y,pred):
    P = []
    R = []
    thresholds = np.arange(0.0,1.0+0.01,0.01)
    for th in thresholds:
        P.append(precision(y,pred,th))
        R.append(recall(y,pred,th))
    plt.figure(figsize=(10,10))
    plt.plot(R,P,linewidth=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall",fontsize=12)
    plt.ylabel("Precision",fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.title("Precision Recall Curve",fontsize=12)
    plt.savefig('Precision_Recall_Curve.png')

def fpr(y,pred,th=0.5):
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    return fp/(tn+fp)

def prevalence(y,pred, th=.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    return (tp+fn)/(tp+fp+tn+fn)

def ppv(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    return tp/(tp+fp)

def npv(y, pred, th=0.5):
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    return tn/(tn+fn)

def get_performance_metrics(y, pred, class_labels, thresholds=[]):
    if len(thresholds) == len(class_labels):
        thresholds = thresholds
    else:
        thresholds = [.5] * len(class_labels)
        for i in range(len(class_labels)):
            threshes = np.arange(.01,.99,.01)
            maxf1 = 0
            bestthresh = 0
            for thresh in threshes:
                testf1 = f1_score(y.iloc[:, i], pred.iloc[:, i]>thresh)
                if testf1 > maxf1:
                    bestthresh = thresh
                    maxf1 = testf1
            thresholds[i] = bestthresh

    columns = ["Labels","Cutoff", "TP","TN","FP","FN","accuracy","ppv", "npv","specificity","sensitivity","AUROC", "F1", "AUPRC","MaxF1"]
    df = pd.DataFrame(columns=columns)

    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        for j in range(1,len(columns)):
            try:
                if columns[j] == 'Cutoff':
                    df.loc[i][j] = round(thresholds[i],3)
                elif columns[j] == 'AUROC':
                    df.loc[i][j] = round(roc_auc_score(y.iloc[:, i], pred.iloc[:, i]), 3)
                elif columns[j] == 'F1':
                    df.loc[i][j] = round(f1_score(y.iloc[:, i], pred.iloc[:, i]>thresholds[i]), 3)
                elif columns[j] == 'AUPRC':
                    df.loc[i][j] = round(average_precision_score(y.iloc[:, i], pred.iloc[:, i]), 3)
                elif columns[j] == 'MaxF1':
                    recall_sig, precision_sig, _ = precision_recall_curve(y.iloc[:,i], pred.iloc[:,i])
                    df.loc[i][j] = max(np.multiply(2, np.divide(np.multiply(precision_sig, recall_sig), np.add(recall_sig, precision_sig))))
                else:
                    df.loc[i][j] = round(globals()[columns[j]](y.iloc[:, i], pred.iloc[:, i], thresholds[i]), 3)
            except:
                df.loc[i][j] = "error"
    df.loc[len(class_labels)] = [""] + [0]*(len(columns) - 1)
    df.loc[len(class_labels)][0] = "Weigted Average"
    total_labels = np.sum(np.sum(y.iloc[:,1:]))
    for j in range(1,len(columns)):
        try:
            sum = 0
            for i in range(1,len(class_labels)):
                sum += df.loc[i][j] *(np.sum(y.iloc[:,i] == 1)/total_labels)
            df.loc[len(class_labels)][j] = round(sum,3)
        except:
            df.loc[len(class_labels)][j] = ""
    return df


# DeLong AUC comparison
# from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]
    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    sample_weight = None
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def calc_auc_ci(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    return auc, ci


def custom_ecg_plot_new(
        ecg,
        sample_rate    = 500,
        lead_index     = None,
        style          = 'bw',
        columns        = 2,
        row_height     = 6,
        show_separate_line  = True,
        label_x_offset         = 0.07,
        label_y_offset         = -0.5,
        fontsize_plot          = 9,
        label_font_style='Nimbus Serif',
        line_width = 0.5,
        ):
    thick_intervals = False
    lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    display_factor = 1

    fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    fig.suptitle('')

    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    elif (style == 'orange'):
        color_major = (0.7,0.17,0.05)
        color_minor = (0.79,0.4,0.06)
        color_line = (0,0,0)
        thick_intervals = True
    elif (style == 'green'):
        color_major = (0.32,0.84,0.04)
        color_minor = (0.9,0.96,0.87)
        color_line = (0,0,0)
    elif (style == 'pink'):
        color_major = (1,0.75,1)
        color_minor = (1,0.5,1)
        color_line = (0,0,0)
    elif (style == 'blank'):
        color_major = (0.99,0.99,0.99)
        color_minor = (0.99,0.99,0.99)
        color_line = (0,0,0)

    ax.set_xticks(np.arange(x_min,x_max,0.2))
    ax.set_yticks(np.arange(y_min,y_max,0.5))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
    ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    if thick_intervals:
        thicker_line_width = 1.0
        for i in np.arange(x_min, x_max, 1.0):
            ax.axvline(x=i, color=(0.67,0.04,0.05), linewidth=thicker_line_width)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)

    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)
                t_lead = lead_order[c * rows + i]
                step = 1.0/sample_rate
                ax.text(x_offset + label_x_offset, y_offset + label_y_offset, lead_index[t_lead], fontsize=fontsize_plot * display_factor, fontname=label_font_style)
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset,
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_line
                )
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
        right=False, labelleft=False, labelbottom=False)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def make_plot(fid, format_ecg):
    formats = ["alternate", "shuffled", "0rhythm", "1rhythm","2rhythm","3rhythm"]
    colors = ["bw", "orange", "green", "pink","blank"]
    label_y_offset = [-0.5,0.5]
    fontsize_plot = [7,9,12,15]
    label_font_style = ['DejaVu Serif','DejaVu Sans','STIXGeneral']
    line_width = [0.3,0.5,0.7,0.9]

    selected_format = random.choice(formats)
    selected_color = random.choice(colors)
    selected_label_y_offset = random.choice(label_y_offset)
    selected_fontsize_plot = random.choice(fontsize_plot)
    selected_label_font_style = random.choice(label_font_style)
    selected_line_width = random.choice(line_width)

    leads_ordered = ['I', 'II', 'III','aVR', 'aVL', 'aVF','V1','V2','V3','V4','V5','V6']
    columns = 4

    try:
        signal = np.load(os.path.join(_SIGNALS_BASE, 'numpy_rp', fid+'.npy'), allow_pickle=True)
        proc_signal = signal[0:5000,:] / 1000
    except:
        signal = np.array(np.load(os.path.join(_SIGNALS_BASE, 'numpy', fid+'.npy'), allow_pickle=True))
        signal = signal.T
        if fid[0]=='2':
            signal = signal.T
            proc_signal = signal[0:5000,:] / 1000
        elif fid[0] =='V':
            proc_signal = signal[0:5000,:] / 1000
        else:
            proc_signal = signal[0:5000,:] / 200

    if format_ecg == '5_0':
        proc_signal = proc_signal[0:2500,:]
        proc_signal = scipy.signal.resample(np.array(proc_signal), 5000)

    proc_signal2 = proc_signal - median_filter(proc_signal,size=(500,1))
    full = proc_signal2.T

    if selected_format == 'alternate':
        col1 = full[0:6, 0:2500]
        col2 = full[6:12,2500:5000]
        newplot1a = np.vstack((col1,col2))
        columns = 2
        lead_index=['I', 'II', 'III','aVR', 'aVL', 'aVF','V1','V2','V3','V4','V5','V6']
    elif selected_format == 'shuffled':
        col1 = full[6:9, 0:1250]
        col1a = full[0:1, 0:1250]
        col2 = full[9:12, 1250:2500]
        col2a = full[0:1, 1250:2500]
        col3 = full[0:3, 2500:3750]
        col3a = full[0:1, 2500:3750]
        col4 = full[3:6, 3750:5000]
        col4a = full[0:1, 3750:5000]
        newplot1a = np.vstack((col1,col1a,col2,col2a,col3,col3a,col4,col4a))
        lead_index=['V1', 'V2', 'V3', 'I','V4', 'V5', 'V6','','I', 'II', 'III', '','aVR', 'aVL', 'aVF','']
    elif selected_format == '0rhythm':
        rhythm_leads = []
        rhythm_lead_idx = []
        lead_index=['I', 'II', 'III']+rhythm_leads+['aVR', 'aVL', 'aVF']+['']*len(rhythm_leads)+ ['V1', 'V2', 'V3']+['']*len(rhythm_leads)+['V4', 'V5', 'V6']+['']*len(rhythm_leads)
        col1 = full[0:3, 0:1250]
        col2 = full[3:6, 1250:2500]
        col3 = full[6:9, 2500:3750]
        col4 = full[9:12, 3750:5000]
        newplot1a = np.vstack((col1,col2,col3,col4))

        rhythm_leads = ['I','V5']
        rhythm_lead_idx = [0,10]
        lead_index=['I', 'II', 'III']+rhythm_leads+['aVR', 'aVL', 'aVF']+['']*len(rhythm_leads)+ ['V1', 'V2', 'V3']+['']*len(rhythm_leads)+['V4', 'V5', 'V6']+['']*len(rhythm_leads)
    elif selected_format == '1rhythm':
        leads = ["I", "II", "V1"]
        selected_lead = random.choice(leads)
        rhythm_leads = [selected_lead]
        rhythm_lead_idx = [leads_ordered.index(selected_lead)]
        lead_index=['I', 'II', 'III']+rhythm_leads+['aVR', 'aVL', 'aVF']+['']*len(rhythm_leads)+ ['V1', 'V2', 'V3']+['']*len(rhythm_leads)+['V4', 'V5', 'V6']+['']*len(rhythm_leads)
        col1 = full[0:3, 0:1250]
        col1a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 0:1250]
        col2 = full[3:6, 1250:2500]
        col2a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 1250:2500]
        col3 = full[6:9, 2500:3750]
        col3a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 2500:3750]
        col4 = full[9:12, 3750:5000]
        col4a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 3750:5000]
        newplot1a = np.vstack((col1,col1a,col2,col2a,col3,col3a,col4,col4a))
    elif selected_format == '2rhythm':
        leads = ["I", "II", "V1"]
        selected_lead1 = random.choice(leads)
        selected_lead2 = random.choice(leads)
        rhythm_leads = [selected_lead1,selected_lead2]
        rhythm_lead_idx = [leads_ordered.index(selected_lead1),leads_ordered.index(selected_lead2)]
        lead_index=['I', 'II', 'III']+rhythm_leads+['aVR', 'aVL', 'aVF']+['']*len(rhythm_leads)+ ['V1', 'V2', 'V3']+['']*len(rhythm_leads)+['V4', 'V5', 'V6']+['']*len(rhythm_leads)
        col1 = full[0:3, 0:1250]
        col1a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 0:1250]
        col1b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 0:1250]
        col2 = full[3:6, 1250:2500]
        col2a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 1250:2500]
        col2b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 1250:2500]
        col3 = full[6:9, 2500:3750]
        col3a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 2500:3750]
        col3b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 2500:3750]
        col4 = full[9:12, 3750:5000]
        col4a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 3750:5000]
        col4b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 3750:5000]
        newplot1a = np.vstack((col1,col1a,col1b,col2,col2a,col2b,col3,col3a,col3b,col4,col4a,col4b))
    elif selected_format == '3rhythm':
        leads = ["I", "II", "V1","V5"]
        selected_lead1 = random.choice(leads)
        selected_lead2 = random.choice(leads)
        selected_lead3 = random.choice(leads)
        rhythm_leads = [selected_lead1,selected_lead2,selected_lead3]
        rhythm_lead_idx = [leads_ordered.index(selected_lead1),leads_ordered.index(selected_lead2),leads_ordered.index(selected_lead3)]
        lead_index=['I', 'II', 'III']+rhythm_leads+['aVR', 'aVL', 'aVF']+['']*len(rhythm_leads)+ ['V1', 'V2', 'V3']+['']*len(rhythm_leads)+['V4', 'V5', 'V6']+['']*len(rhythm_leads)
        col1 = full[0:3, 0:1250]
        col1a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 0:1250]
        col1b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 0:1250]
        col1c = full[rhythm_lead_idx[2]:rhythm_lead_idx[2]+1, 0:1250]
        col2 = full[3:6, 1250:2500]
        col2a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 1250:2500]
        col2b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 1250:2500]
        col2c = full[rhythm_lead_idx[2]:rhythm_lead_idx[2]+1, 1250:2500]
        col3 = full[6:9, 2500:3750]
        col3a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 2500:3750]
        col3b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 2500:3750]
        col3c = full[rhythm_lead_idx[2]:rhythm_lead_idx[2]+1, 2500:3750]
        col4 = full[9:12, 3750:5000]
        col4a = full[rhythm_lead_idx[0]:rhythm_lead_idx[0]+1, 3750:5000]
        col4b = full[rhythm_lead_idx[1]:rhythm_lead_idx[1]+1, 3750:5000]
        col4c = full[rhythm_lead_idx[2]:rhythm_lead_idx[2]+1, 3750:5000]
        newplot1a = np.vstack((col1,col1a,col1b,col1c,col2,col2a,col2b,col2c,col3,col3a,col3b,col3c,col4,col4a,col4b,col4c))

    get_data = custom_ecg_plot_new(newplot1a, sample_rate=500, show_separate_line=False,
        columns=columns, label_x_offset=0.07, label_y_offset=selected_label_y_offset,
        fontsize_plot=selected_fontsize_plot, label_font_style=selected_label_font_style,
        line_width=selected_line_width, lead_index=lead_index, style=selected_color)

    return get_data


class Train_New(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size):
        self.df = df
        self.bsz = batch_size
        class_names = ['Dermatophytosis']
        self.labels = self.df[class_names].values
        self.im_list = self.df['fileID'].tolist()
        self.format_list = self.df['format_new'].to_list()
        self.idxs = [i for i in range(len(self.im_list))]

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def get_batch_labels(self, idx):
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        return np.array([resize(ndimage.rotate(resize(np.array(Image.fromarray(make_plot(self.im_list[index],self.format_list[index])).convert('L').convert('RGB')), (300,300)), np.random.uniform(-10,10,1)[0],reshape=True,mode='nearest'),(300,300)) for index in self.idxs[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


class DataSequenceRAM(tf.keras.utils.Sequence):
    """
    Keras Sequence for training with images already loaded into RAM (as numpy arrays).
    Expects a column 'image_array' with shape (300, 300, 3) in the input dataframe.
    """
    def __init__(self, df, batch_size, label):
        self.df = df.reset_index(drop=True)
        self.images = df['image_array'].values
        self.labels = df[[label]].values.astype(np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.stack(batch_x), batch_y


class DataSequenceAugRAM(tf.keras.utils.Sequence):
    """
    Like DataSequenceRAM but applies random rotation ±max_angle degrees per batch.
    Uses TF RandomRotation (faster than scipy ndimage, no Python GIL bottleneck).
    Expects a column 'image_array' with shape (300, 300, 3) in the input dataframe.
    Images should already be normalised to [0, 1].
    """
    def __init__(self, df, batch_size, label, max_angle=10):
        self.df = df.reset_index(drop=True)
        self.images = df['image_array'].values
        self.labels = df[[label]].values.astype(np.float32)
        self.batch_size = batch_size
        # factor = fraction of full rotation; ±max_angle degrees = max_angle/360
        # dtype='float32': ImageProjectiveTransformV3 rejects bfloat16 (mixed-precision global policy)
        self.aug = tf.keras.layers.RandomRotation(
            factor=max_angle / 360.0, fill_mode='nearest', seed=None, dtype='float32'
        )

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = np.stack(self.images[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Run rotation on CPU so it doesn't contend with training on the GPU stream
        with tf.device('/CPU:0'):
            augmented = self.aug(batch_x.astype(np.float32), training=True).numpy()
        return augmented.astype(np.float32), batch_y


class DataSequenceTrain_RAM(tf.keras.utils.Sequence):
    """
    Loads ECG images into RAM using multithreading.
    During training, fetches batches and applies augmentations (rotation + resize).
    """
    def __init__(self, df, batch_size, label):
        self.df = df
        self.bsz = batch_size
        self.labels = self.df[label].values
        self.im_list = self.df['fileID'].tolist()
        self.format_list = self.df['format_new'].tolist()
        self.idxs = list(range(len(self.im_list)))

        def load_single_image(args):
            fid, fmt = args
            arr = make_plot(fid, fmt)
            img = Image.fromarray(arr).convert('L').convert('RGB')
            img = img.resize((300, 300))
            return np.array(img)

        with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            tasks = zip(self.im_list, self.format_list)
            self.images = list(tqdm(executor.map(load_single_image, tasks),
                                    total=len(self.im_list),
                                    desc="Preloading images into RAM"))

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def get_batch_labels(self, idx):
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        batch_imgs = []
        for index in self.idxs[idx * self.bsz: (idx + 1) * self.bsz]:
            base = self.images[index]
            angle = float(np.random.uniform(-10, 10, 1)[0])
            rot = ndimage.rotate(base, angle, reshape=True, mode='nearest')
            out = resize(rot, (300, 300))
            batch_imgs.append(out)
        return np.array(batch_imgs)

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


def save_all_images(df, output_dir, num_workers=None, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"number of workers = {num_workers} (overwrite={overwrite})")

    func = partial(save_image_from_row, output_dir=output_dir, overwrite=overwrite)
    statuses = []
    with Pool(num_workers) as pool:
        for status in tqdm(pool.imap(func, df.to_dict('records')), total=len(df), desc="Saving ECG Images"):
            statuses.append(status)

    from collections import Counter
    return dict(Counter(statuses))


def save_image_from_row(row, output_dir, overwrite=False):
    fid = row['fileID']
    fmt = row.get('format_new', None)
    os.makedirs(output_dir, exist_ok=True)
    filename = fid.split("/")[-1] + ".png"
    out_path = os.path.join(output_dir, filename)

    if os.path.exists(out_path) and not overwrite:
        return 'skipped'
    try:
        img = make_plot(fid, fmt)
        img_pil = Image.fromarray(img).convert('RGB').resize((300, 300))
        img_pil.save(out_path, format='PNG')
        return 'saved'
    except Exception as e:
        print(f"Failed {fid}: {e}")
        return 'error'


def load_image_from_disk(fid, image_dir):
    filename = os.path.basename(fid)
    path = os.path.join(image_dir, f"{filename}.png")
    try:
        img = Image.open(path).convert('RGB').resize((300, 300))
        img = np.array(img).astype(np.float32) / 255.0
        return img
    except Exception as e:
        print(f"Failed to load {fid}: {e}")
        return None


def load_images_parallel(file_ids, image_dir, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    file_ids = list(file_ids)
    load_fn = partial(load_image_from_disk, image_dir=image_dir)

    print(f"Starting parallel loading of {len(file_ids)} images with {num_workers} workers...")

    pool = Pool(num_workers)
    try:
        images = list(tqdm(pool.imap(load_fn, file_ids), total=len(file_ids), desc="Loading images"))
    finally:
        pool.close()
        pool.join()

    print("Pool closed and joined.")
    return dict(zip(file_ids, images))


def loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    weights = np.array([[1.3, 0.77]])
    return K.mean((weights[:, 1] ** (1 - y_true)) * (weights[:, 0] ** y_true) * K.binary_crossentropy(y_true, y_pred), axis=-1)


def run_model_prediction(model_path, test_df, output_csv, outcome, image_path, batch_size=90):
    if outcome not in test_df.columns:
        test_df[outcome] = True

    print(f"Outcome column '{outcome}' added. Starting prediction...")

    test_sequence = DataSequenceTest(df=test_df, batch_size=batch_size, mode="image", class_names=outcome)

    num_workers = min(multiprocessing.cpu_count(), 32)
    print(f"Using {num_workers} workers for prediction...")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = load_model(model_path, custom_objects={"loss_fn": loss_fn})

    predictions = model.predict(
        test_sequence,
        max_queue_size=30,
        use_multiprocessing=True,
        workers=num_workers,
        verbose=1
    )

    test_df[f'preds_{outcome}'] = predictions
    test_df.to_csv(output_csv, index=False)
    return test_df
