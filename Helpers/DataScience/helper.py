"""
Provides helper functions for data science projects
"""

import os
import shutil

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python as tfp
import tensorflow.python.keras.models as models
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.callbacks as callbacks
import tensorflow.python.keras.optimizers as optimizers
import tensorflow_hub as hub

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.optimizer_v2.adam import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_loss_metrics(history, metrics, validation_metrics=True, all_in_one=False, figsize=(10, 6)):
    '''
    Plots loss and metric curves from the history object

    Args:
        history: Tensorflow History object.
        metrics: Metrics to plot curves for other than loss.
        validation_metrics: {default: True} If False, will not plot metrics for validation data.
        all_in_one: {default: False} If True, will plot all curves on a single figure.
        figsize: A tuple holding the figure size to plot on.
    '''

    history_df = pd.DataFrame(history.history)
    epochs = range(len(history_df))

    plt.figure(figsize=figsize)
    history_df['loss'].plot(x=epochs)
    if validation_metrics:
        history_df['val_loss'].plot(x=epochs)
    plt.title('Loss Curve')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()

    for metric in metrics:
        if not all_in_one:
            plt.figure(figsize=figsize)
        history_df[metric].plot(x=epochs)
        if validation_metrics:
            history_df['val_' + metric].plot(x=epochs)
        plt.title(metric[0].upper() + metric[1:] + ' Curve')
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.legend()
