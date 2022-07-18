from google.colab import drive
drive.mount('/content/drive')


import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from k_fold_report_utils import report
from plot_utils import plot_confusion_matrix, plot_roc, train_curves

from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model

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

from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientb2

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

from tensorflow.keras.applications.mobilenet_v2  import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2


from tensorflow.keras import Model, layers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Average, Maximum, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization


from numpy.testing import assert_allclose

from keras.utils.vis_utils import plot_model


import gc


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)


AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH_1 = "gs://" # large-covid19-ct-slice-dataset
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [224, 224]
IMAGE_SHAPE = 224
EPOCHS = 30

filenames = tf.io.gfile.glob(str(GCS_PATH_1 + "/curated_data/curated_data/*/*.png"))
exp_filenames, test_filenames = train_test_split(filenames, test_size=0.05,  random_state = 7)

print(len(filenames))
print(len(exp_filenames))
print(len(test_filenames))

plt.rcParams.update({'legend.fontsize': 10,
                    'axes.labelsize': 14, 
                    'axes.titlesize': 14,
                    'xtick.labelsize': 14,
                    'ytick.labelsize': 14})


def build_deep_cnn_model():

    deep_cnn_base_model = EfficientNetB2(    # VGG19(
                                             # ResNet101(
                                             # InceptionV3(
                                             # ResNet152V2(                                                
                                             # InceptionResNetV2(
                                             # DenseNet201(
                                             # Xception(
                                             # MobileNetV2(
                                    input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                    include_top=False, 
                                    weights="imagenet")

    deep_cnn_base_model.trainable = True

    inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    deep_cnn_preprocess_input = preprocess_input_efficientb2(inputs) # preprocess_input_vgg19(inputs)
                                                                     # preprocess_input_resnet101(inputs)
                                                                     # preprocess_input_inceptionv3(inputs)
                                                                     # preprocess_input_resnet152v2(inputs)
                                                                     # preprocess_input_inception_resnetv2(inputs)
                                                                     # preprocess_input_inception_resnetv2(inputs)
                                                                     # preprocess_input_densenet201(inputs)
                                                                     # preprocess_input_xception(inputs)
                                                                     # preprocess_input_mobilenet_v2(inputs)
    deep_cnn_output = deep_cnn_base_model(deep_cnn_preprocess_input)
    x = Flatten()(deep_cnn_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    deep_cnn_model = Model(inputs, prediction)

    deep_cnn_model._name = "EfficientNetB2_Model"   # "VGG19_Model"
                                                    # "Resnet101_Model"
                                                    # "InceptionV3_Model"
                                                    # "ResNet152V2_Model"
                                                    # "Inception_ResnetV2_Model"
                                                    # "DenseNet201_Model"
                                                    # "Xception_Model"
                                                    # "MobileNetV2_Model"

    return deep_cnn_model

def deep_cnn_model():
    with strategy.scope():
        deep_cnn_model = build_deep_cnn_model()
                
        deep_cnn_model.compile(optimizer = "adam",
                                loss = "categorical_crossentropy",
                                metrics=["accuracy"])

    return deep_cnn_model   

model = deep_cnn_model()
init_weights = model.get_weights()
report(model, init_weights, exp_filenames, test_filenames, epochs=EPOCHS, autotune=AUTOTUNE, batch_size=BATCH_SIZE)