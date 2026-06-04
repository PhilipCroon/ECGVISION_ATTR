import warnings
import csv

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from PIL import ImageFile
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB3
import tensorflow.keras.backend as K
from keras.callbacks import CSVLogger
from utils import *


temperature = 0.1
width = 128


class ContrastiveModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_gradients = 16
        self.temperature = temperature
        self.encoder0 = EfficientNetB3(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        x = GlobalAveragePooling2D(name='avg_pool')(self.encoder0.output)
        self.encoder = Model(self.encoder0.input, x)
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(1536,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )

        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        self.n_acum_step.assign_add(1)
        unlabeled_images_1, unlabeled_images_2 = data
        augmented_images_1 = unlabeled_images_1
        augmented_images_2 = unlabeled_images_2

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )

        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        self.contrastive_optimizer.apply_gradients(
            zip(
                self.gradient_accumulation,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


def save_csv(filename, data):
    data = [(i, r) for i, r in enumerate(data)]
    data.insert(0, ('ID', 'TARGET'))
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)
