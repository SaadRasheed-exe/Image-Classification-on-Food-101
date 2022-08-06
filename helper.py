"""
Provides helper functions for data science projects
"""

import os
import shutil
import random
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
        history: Tensorflow History object.\n
        metrics: Metrics to plot curves for other than loss.\n
        validation_metrics: {default: True} If False, will not plot metrics for validation data.\n
        all_in_one: {default: False} If True, will plot all curves on a single figure.\n
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


def view_random_image(data_dir, labels, split='train', figsize=(10, 6), n_samples=1):
    '''
    Finds and displays random images from a preprocessed image dataset.

    Args:
        data_dir: Path to the dataset directory.\n
        labels: List of labels in the dataset.\n
        split: {default: 'train'} The set from which to display the image.\n
        figsize: Size of each image.\n
        n_samples: Number of images to show.
    '''

    random_label = []
    path_to_random_image = []

    for sample in range(n_samples):
        random_label.append(labels[random.randint(0, len(labels) - 1)])
        path_to_random_label = os.path.join(
            data_dir, split, random_label[sample])
        path_to_random_image.append(os.path.join(
            path_to_random_label, np.random.choice(os.listdir(path_to_random_label))))

    for sample in range(n_samples):
        plt.figure(figsize=figsize)
        plt.imshow(mpimg.imread(path_to_random_image[sample]))
        plt.axis(False)
        plt.title(random_label[sample])
