"""
Model definition for ECGVISION-ATTR: EfficientNetB3 transfer head on top of a
contrastive-pretrained encoder. Ported from the existing training script.

This is the layer to ALTER when experimenting (head depth/width, dropout,
unfreeze schedule, class weights, optimizer/LR). Defaults below reproduce the
current production model exactly.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
import tensorflow.keras.backend as K

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

# ContrastiveModel etc. live in the shared image-models repo.
sys.path.append(project.ecg_image_models_path)
from model_helpers import *  # noqa: F401,F403  (ContrastiveModel)

# === Alteration knobs (defaults = current production model) ===
CONTRASTIVE_WEIGHTS = project.contrastive_weights
CLASS_WEIGHTS = np.array([[1.3, 0.77]])   # [pos, neg]
HEAD_UNITS = (64, 32)
HEAD_DROPOUT = 0.2
LR_FROZEN = 1e-3
# Plain float (NOT a schedule) so ReduceLROnPlateau can drive it. Start hot at 1e-4
# (prev 2e-5 was effectively constant + underfit); plateau callback drops it when
# val_auroc stalls. See train.py unfrozen callbacks.
LR_UNFROZEN = 1e-4


# Loss selector (env-configurable so it's one knob per run; default = current wbce):
#   LOSS=wbce  -> class-weighted binary cross-entropy (production default)
#   LOSS=focal -> binary focal loss, FOCAL_GAMMA (default 2.0), FOCAL_ALPHA (weight on
#                 the positive/amyloid term, default 0.75 to upweight the rare class)
LOSS = os.getenv('LOSS', 'wbce').lower()
FOCAL_GAMMA = float(os.getenv('FOCAL_GAMMA', '2.0'))
FOCAL_ALPHA = float(os.getenv('FOCAL_ALPHA', '0.75'))


def loss_fn(y_true, y_pred):
    """Dispatch: class-weighted BCE (default) or binary focal loss (LOSS=focal).

    Registered as 'loss_fn' for keras load_model(custom_objects); the dispatch is
    internal so saved models keep loading regardless of which loss trained them.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if LOSS == 'focal':
        eps = K.epsilon()
        p = K.clip(y_pred, eps, 1.0 - eps)
        ce_pos = -y_true * K.log(p)
        ce_neg = -(1.0 - y_true) * K.log(1.0 - p)
        fl = (FOCAL_ALPHA * K.pow(1.0 - p, FOCAL_GAMMA) * ce_pos
              + (1.0 - FOCAL_ALPHA) * K.pow(p, FOCAL_GAMMA) * ce_neg)
        return K.mean(fl, axis=-1)
    # default: class-weighted binary cross-entropy
    w = CLASS_WEIGHTS
    return K.mean(
        (w[:, 1] ** (1 - y_true)) * (w[:, 0] ** (y_true))
        * K.binary_crossentropy(y_true, y_pred), axis=-1)


def _metrics():
    return [tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(curve='ROC', name='auroc'),
            tf.keras.metrics.AUC(curve='PR', name='auprc')]


def build_transfer_model():
    """EfficientNetB3 encoder (contrastive init, frozen) + dense head."""
    # Checkpoint was saved in float32; build encoder under float32 to avoid
    # dtype mismatch when mixed_bfloat16 is the global policy.
    _policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float32')
    pretraining_model = ContrastiveModel()
    pretraining_model.built = True
    pretraining_model.load_weights(CONTRASTIVE_WEIGHTS)
    pretraining_model.trainable = False
    tf.keras.mixed_precision.set_global_policy(_policy)

    x = BatchNormalization()(pretraining_model.encoder.output)
    x = Dropout(HEAD_DROPOUT)(x)
    for units in HEAD_UNITS:
        x = Dense(units, activation='relu')(x)
        x = Dropout(HEAD_DROPOUT)(x)
    x = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(pretraining_model.encoder.input, x)
    opt = tf.keras.optimizers.Adam(learning_rate=LR_FROZEN, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=_metrics())
    return model


def unfreeze_model(model):
    """Unfreeze all but BatchNorm, recompile at the fine-tune LR."""
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=LR_UNFROZEN, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=_metrics())


def gc_callback():
    import gc

    class _GC(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
    return _GC()
