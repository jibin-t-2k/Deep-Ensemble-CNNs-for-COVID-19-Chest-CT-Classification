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



def build_nn_ensemble_model():

    os.chdir("/content/drive/MyDrive/saved_models")
    resnet101_model = load_model("Resnet101_Model.h5")
    densenet201_model = load_model("DenseNet201_Model.h5")
    efficientb7_model = load_model("EfficientNetB7_Model.h5")
    xception_model = load_model("Xception_Model.h5")
    os.chdir("/content/")
    
    resnet101_model._name = "ResNet101"
    resnet101_model.trainable = False
    resnet101_base = resnet101_model.get_layer(name = "resnet101")

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_base = densenet201_model.get_layer(name = "densenet201")

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_base = efficientb7_model.get_layer(name = "efficientnetb7")

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_base = xception_model.get_layer(name = "xception")

    inputs = Input(shape=(224, 224, 3))

    resnet101_preprocess_input = preprocess_input_resnet101(inputs)
    resnet101_base_output = resnet101_base(resnet101_preprocess_input)
    resnet101_output = GlobalAveragePooling2D()(resnet101_base_output)
    resnet101_output = BatchNormalization()(resnet101_output)

    densenet201_preprocess_input = preprocess_input_densenet201(inputs)
    densenet201_base_output = densenet201_base(densenet201_preprocess_input)
    densenet201_output = GlobalAveragePooling2D()(densenet201_base_output)
    densenet201_output = BatchNormalization()(densenet201_output)

    efficientb2_preprocess_input = preprocess_input_efficientb2(inputs)
    efficientb2_base_output = efficientb7_base(efficientb2_preprocess_input)
    efficientb2_output = GlobalAveragePooling2D()(efficientb2_base_output)
    efficientb2_output = BatchNormalization()(efficientb2_output)

    xception_preprocess_input = preprocess_input_xception(inputs)
    xception_base_output = xception_base(xception_preprocess_input)
    xception_output = GlobalAveragePooling2D()(xception_base_output)
    xception_output = BatchNormalization()(xception_output)

    concat_output = Concatenate(axis=-1)([
                                        resnet101_output,
                                        densenet201_output,
                                        efficientb2_output,
                                        xception_output
                                        ])

    x = Dense(1024,activation ="relu")(concat_output)
    x = Dropout(0.5)(x)
    x = Dense(256,activation ="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64,activation ="relu")(x)
    prediction = Dense(3, activation="softmax")(x)

    ensemble_model = Model(inputs, prediction)

    ensemble_model._name = "NN_Ensemble_Trained_CNN_Model"

    return ensemble_model

def nn_ensemble_model():
    with strategy.scope():
        nn_ensemble_model = build_nn_ensemble_model()

        nn_ensemble_model.compile(optimizer = "adam",
                                    loss = "categorical_crossentropy",
                                    metrics=["accuracy"])

    return nn_ensemble_model   

model = nn_ensemble_model()
init_weights = model.get_weights()
report(model, init_weights, exp_filenames, test_filenames, epochs=EPOCHS, autotune=AUTOTUNE, batch_size=BATCH_SIZE)



def build_conv_ensemble_model():

    os.chdir("/content/drive/MyDrive/saved_models")
    resnet101_model = load_model("Resnet101_Model.h5")
    densenet201_model = load_model("DenseNet201_Model.h5")
    efficientb7_model = load_model("EfficientNetB7_Model.h5")
    xception_model = load_model("Xception_Model.h5")
    os.chdir("/content/")
    
    resnet101_model._name = "ResNet101"
    resnet101_model.trainable = False
    resnet101_base = resnet101_model.get_layer(name = "resnet101")

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_base = densenet201_model.get_layer(name = "densenet201")

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_base = efficientb7_model.get_layer(name = "efficientnetb7")

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_base = xception_model.get_layer(name = "xception")

    inputs = Input(shape=(224, 224, 3))

    resnet101_preprocess_input = preprocess_input_resnet101(inputs)
    resnet101_base_output = resnet101_base(resnet101_preprocess_input)
    resnet101_output = BatchNormalization()(resnet101_base_output)

    densenet201_preprocess_input = preprocess_input_densenet201(inputs)
    densenet201_base_output = densenet201_base(densenet201_preprocess_input)
    densenet201_output = BatchNormalization()(densenet201_base_output)

    efficientb2_preprocess_input = preprocess_input_efficientb2(inputs)
    efficientb2_base_output = efficientb7_base(efficientb2_preprocess_input)
    efficientb2_output = BatchNormalization()(efficientb2_base_output)

    xception_preprocess_input = preprocess_input_xception(inputs)
    xception_base_output = xception_base(xception_preprocess_input)
    xception_output = BatchNormalization()(xception_base_output)

    concat_output = Concatenate(axis=-1)([
                                        resnet101_output,
                                        densenet201_output,
                                        efficientb2_output,
                                        xception_output
                                        ])

    x = Conv2D(2048, kernel_size=(2,2), strides=1, activation="relu")(concat_output)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(512, kernel_size=(2,2), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(64, kernel_size=(2,2), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Flatten()(x)
    prediction = Dense(3, activation="softmax")(x)

    ensemble_model = Model(inputs, prediction)

    ensemble_model._name = "Convolution_Ensemble_Trained_CNNs_Model"

    return ensemble_model


def conv_ensemble_model():
    with strategy.scope():
        conv_ensemble_model = build_conv_ensemble_model()
          
        conv_ensemble_model.compile(optimizer = "adam",
                                    loss = "categorical_crossentropy",
                                    metrics=["accuracy"])

    return conv_ensemble_model   

model = conv_ensemble_model()
init_weights = model.get_weights()
report(model, init_weights, exp_filenames, test_filenames, epochs=EPOCHS, autotune=AUTOTUNE, batch_size=BATCH_SIZE)
