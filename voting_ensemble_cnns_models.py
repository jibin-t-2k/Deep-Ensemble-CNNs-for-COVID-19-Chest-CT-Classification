from google.colab import drive
drive.mount('/content/drive')

import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from plot_utils import plot_confusion_matrix, plot_roc

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnetv2

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet101

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet152v2

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet201

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientb7

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

from tensorflow.keras.applications.mobilenet_v2  import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2

from tensorflow.keras import Model, layers, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Average, Maximum, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization, LayerNormalization, Activation

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from k_fold_report_utils import total_eval, test_eval

from numpy.testing import assert_allclose

from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

import gc

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)
    
print(tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH_1 = "gs://" # large-covid19-ct-slice-dataset
GCS_PATH_2 = "gs://" # covidxct
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [224, 224]
IMAGE_SHAPE = 224
EPOCHS = 30

filenames = tf.io.gfile.glob(str(GCS_PATH_1 + "/curated_data/curated_data/*/*.png"))

exp_filenames, test_filenames = train_test_split(filenames, test_size=0.05,  random_state = 7)

print(len(filenames))
print(len(exp_filenames))
print(len(test_filenames))


def build_average_ensemble_model():

    os.chdir("/content/drive/MyDrive/saved_models")
    resnet101_model = load_model("Resnet101_Model.h5")
    densenet201_model = load_model("DenseNet201_Model.h5")
    efficientb7_model = load_model("EfficientNetB7_Model.h5")
    xception_model = load_model("Xception_Model.h5")
    os.chdir("/content/")
    
    inputs = Input(shape=(224, 224, 3))

    resnet101_model._name = "ResNet101"
    resnet101_model.trainable = False
    resnet101_pred = resnet101_model(inputs)

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_pred = densenet201_model(inputs)

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_pred = efficientb7_model(inputs)

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_pred = xception_model(inputs)

    avg_output = Average()([
                            resnet101_pred,
                            densenet201_pred,
                            efficientb7_pred,
                            xception_pred,
                            ])

    ensemble_model = Model(inputs, avg_output)

    ensemble_model._name = "Average_Ensemble_Trained_CNNs_Model"

    return ensemble_model

def average_ensemble_model():
    with strategy.scope():
        average_ensemble_model = build_average_ensemble_model()
          
        average_ensemble_model.compile(optimizer = "adam",
                                      loss = "categorical_crossentropy",
                                      metrics=["accuracy"])

    return average_ensemble_model   

average_ensemble_model = average_ensemble_model()
test_eval(average_ensemble_model, test_filenames, autotune=AUTOTUNE, batch_size=BATCH_SIZE)
total_eval(average_ensemble_model, filenames, autotune=AUTOTUNE, batch_size=BATCH_SIZE)


def build_majority_ensemble_model():

    os.chdir("/content/drive/MyDrive/saved_models")
    resnet101_model = load_model("Resnet101_Model.h5")
    densenet201_model = load_model("DenseNet201_Model.h5")
    efficientb7_model = load_model("EfficientNetB7_Model.h5")
    xception_model = load_model("Xception_Model.h5")
    os.chdir("/content/")
    
    inputs = Input(shape=(224, 224, 3))

    resnet101_model._name = "ResNet101"
    resnet101_model.trainable = False
    resnet101_pred = resnet101_model(inputs)

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_pred = densenet201_model(inputs)

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_pred = efficientb7_model(inputs)

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_pred = xception_model(inputs)

    majority_output = Maximum()([
                                resnet101_pred,
                                densenet201_pred,
                                efficientb7_pred,
                                xception_pred,
                                ])

    ensemble_model = Model(inputs, majority_output)

    ensemble_model._name = "Majority_Ensemble_Trained_CNNs_Model"

    return ensemble_model

def majority_ensemble_model():
    with strategy.scope():
        majority_ensemble_model = build_majority_ensemble_model()
          
        majority_ensemble_model.compile(optimizer = "adam",
                                        loss = "categorical_crossentropy",
                                        metrics=["accuracy"])

    return majority_ensemble_model   

majority_ensemble_model = majority_ensemble_model()
test_eval(majority_ensemble_model, test_filenames, autotune=AUTOTUNE, batch_size=BATCH_SIZE)
total_eval(majority_ensemble_model, filenames, autotune=AUTOTUNE, batch_size=BATCH_SIZE)