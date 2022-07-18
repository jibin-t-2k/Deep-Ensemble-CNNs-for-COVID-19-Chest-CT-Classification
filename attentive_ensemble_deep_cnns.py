from google.colab import drive
drive.mount('/content/drive')


import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model

from k_fold_report_utils import report
from plot_utils import plot_confusion_matrix, plot_roc, train_curves

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Concatenate, Conv2D, Flatten, Average, Maximum, MaxPooling2D, MultiHeadAttention,
                                     Dropout, Activation, GlobalAveragePooling2D, BatchNormalization, LayerNormalization)


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



def build_attentive_ensemble_model():

    os.chdir("/content/drive/MyDrive/saved_models")
    resnet152v2_model = load_model("ResNet152V2_Model.h5")
    densenet201_model = load_model("DenseNet201_Model.h5")
    efficientb7_model = load_model("EfficientNetB7_Model.h5")
    xception_model = load_model("Xception_Model.h5")
    os.chdir("/content/")
    
    resnet152v2_model._name = "ResNet152V2"
    resnet152v2_model.trainable = False
    resnet152v2_base = resnet152v2_model.get_layer(name = "resnet152v2")

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_base = densenet201_model.get_layer(name = "densenet201")

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_base = efficientb7_model.get_layer(name = "efficientnetb7")

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_base = xception_model.get_layer(name = "xception")

    del resnet152v2_model
    del densenet201_model
    del efficientb7_model
    del xception_model
    for k in range(5):
        gc.collect()

    inputs = Input(shape=(224, 224, 3))

    resnet152v2_preprocess_input = preprocess_input_resnet152v2(inputs)
    resnet152v2_base_output = resnet152v2_base(resnet152v2_preprocess_input)
    resnet152v2_output = Conv2D(512, 1, padding='same', use_bias=False)(resnet152v2_base_output)
    resnet152v2_output = BatchNormalization()(resnet152v2_output)
    resnet152v2_output = Activation("relu")(resnet152v2_output)


    densenet201_preprocess_input = preprocess_input_densenet201(inputs)
    densenet201_base_output = densenet201_base(densenet201_preprocess_input)
    densenet201_output = Conv2D(512, 1, padding='same', use_bias=False)(densenet201_base_output)
    densenet201_output = BatchNormalization()(densenet201_output)
    densenet201_output = Activation("relu")(densenet201_output)

    efficientb7_preprocess_input = preprocess_input_efficientb7(inputs)
    efficientb7_base_output = efficientb7_base(efficientb7_preprocess_input)
    efficientb7_output = Conv2D(512, 1, padding='same', use_bias=False)(efficientb7_base_output)
    efficientb7_output = BatchNormalization()(efficientb7_output)
    efficientb7_output = Activation("relu")(efficientb7_output)


    xception_preprocess_input = preprocess_input_xception(inputs)
    xception_base_output = xception_base(xception_preprocess_input)
    xception_output = Conv2D(512, 1, padding='same', use_bias=False)(xception_base_output)
    xception_output = BatchNormalization()(xception_output)
    xception_output = Activation("relu")(xception_output)

    stack = tf.stack([
            resnet152v2_output,
            densenet201_output,
            efficientb7_output,
            xception_output
            ], axis=1)

    x_m1 = MultiHeadAttention(num_heads=4, key_dim=512, dropout=0.2)(stack, stack)
    x1 = Dropout(0.3)(x_m1)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + stack)

    x1 = Conv2D(filters=256, kernel_size=3, activation='relu')(x1)
    x1 = Dropout(0.3)(x1)
    i1 = Conv2D(filters=256, kernel_size=3, activation='relu')(stack)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + i1)

    x_m2 = MultiHeadAttention(num_heads=4, key_dim=256, dropout=0.2)(x1, x1)
    x2 = Dropout(0.3)(x_m2)
    x2 = LayerNormalization(epsilon=1e-6)(x2 + x1)

    x2 = Conv2D(filters=64, kernel_size=3, activation='relu')(x2)
    x2 = Dropout(0.3)(x2)
    i2 = Conv2D(filters=64, kernel_size=3, activation='relu')(x1)
    x2 = LayerNormalization(epsilon=1e-6)(x2 + i2)
            
    x_m3 = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x2, x2)
    x3 = Dropout(0.2)(x_m3)
    x3 = LayerNormalization(epsilon=1e-6)(x3 + x2)

    x3 = Conv2D(filters=16, kernel_size=3, activation='relu')(x3)
    x3 = Dropout(0.2)(x3)
    i3 = Conv2D(filters=16, kernel_size=3, activation='relu')(x2)
    x3 = LayerNormalization(epsilon=1e-6)(x3 + i3)

    x = Flatten()(x3)
    prediction = Dense(3, activation="softmax")(x)

    attent_ensemble_model = Model(inputs, prediction)

    attent_ensemble_model._name = "Attention_Ensemble_Trained_CNNs_Model"

    return attent_ensemble_model   



model = build_attentive_ensemble_model()
init_weights = model.get_weights()
report(model, init_weights, exp_filenames, test_filenames, epochs=EPOCHS, autotune=AUTOTUNE, batch_size=BATCH_SIZE)
