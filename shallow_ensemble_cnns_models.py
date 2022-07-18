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

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnetv2

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet101

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet201

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet152v2

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientb7

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

from tensorflow.keras.applications.mobilenet_v2  import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2

from tensorflow.keras import Model, layers, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Average, Maximum, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from plot_utils import plot_confusion_matrix, plot_roc, train_curves

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

from numpy.testing import assert_allclose

from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pickle
from joblib import dump, load
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
# filenames_2 = tf.io.gfile.glob(str(GCS_PATH_2 + "/2A_images/*.png"))
# filenames = filenames + filenames_2
exp_filenames, test_filenames = train_test_split(filenames, test_size=0.05,  random_state = 7)

print(len(filenames))
print(len(exp_filenames))
print(len(test_filenames))

def get_gt(file_path):
    CLASSES = ["1NonCOVID", "2COVID", "3CAP"]
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASSES #return ground truth

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    gt = get_gt(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, gt

def prepare_for_training(ds, cache=True, shuffle = True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don"t
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size = shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size = AUTOTUNE)

    return ds


def build_stacked_ensemble_model():

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

    inputs = Input(shape=(224, 224, 3))

    resnet152v2_preprocess_input = preprocess_input_resnet152v2(inputs)
    resnet152v2_base_output = resnet152v2_base(resnet152v2_preprocess_input)
    resnet152v2_output = GlobalAveragePooling2D()(resnet152v2_base_output)
    resnet152v2_output = BatchNormalization()(resnet152v2_output)

    densenet201_preprocess_input = preprocess_input_densenet201(inputs)
    densenet201_base_output = densenet201_base(densenet201_preprocess_input)
    densenet201_output = GlobalAveragePooling2D()(densenet201_base_output)
    densenet201_output = BatchNormalization()(densenet201_output)

    efficientb7_preprocess_input = preprocess_input_efficientb7(inputs)
    efficientb7_base_output = efficientb7_base(efficientb7_preprocess_input)
    efficientb7_output = GlobalAveragePooling2D()(efficientb7_base_output)
    efficientb7_output = BatchNormalization()(efficientb7_output)

    xception_preprocess_input = preprocess_input_xception(inputs)
    xception_base_output = xception_base(xception_preprocess_input)
    xception_output = GlobalAveragePooling2D()(xception_base_output)
    xception_output = BatchNormalization()(xception_output)

    concat_output = Concatenate(axis=-1)([
                                        resnet152v2_output,
                                        densenet201_output,
                                        efficientb7_output,
                                        xception_output
                                        ])
    
    ensemble_model = Model(inputs, concat_output)

    ensemble_model.compile(optimizer = "adam",
                    loss = "categorical_crossentropy",
                    metrics=["accuracy"])

    ensemble_model._name = "Stacked_Ensemble_Model"

    return ensemble_model


with strategy.scope():
    stack_ensemble_model = build_stacked_ensemble_model()

train_list_ds = tf.data.Dataset.from_tensor_slices(exp_filenames)
test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)

TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
TEST_IMG_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()

train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = prepare_for_training(train_ds, shuffle=False)
test_ds = prepare_for_training(test_ds, shuffle=False)

train_stack_output = stack_ensemble_model.predict(train_ds, steps = (TRAIN_IMG_COUNT // (BATCH_SIZE//1.5)), verbose = 1)
test_stack_output = stack_ensemble_model.predict(test_ds, steps = (TEST_IMG_COUNT // (BATCH_SIZE//1.5)), verbose = 1)

np.save('/content/drive/MyDrive/stacked_output/train_stack_output.npy', train_stack_output[:TRAIN_IMG_COUNT])
np.save('/content/drive/MyDrive/stacked_output/test_stack_output.npy', test_stack_output[:TEST_IMG_COUNT])



exp_stack_output = np.load('/content/train_stack_output.npy')#, mmap_mode="r+")
exp_y = np.load('/content/train_y.npy')#, mmap_mode="r+")
test_stack_output = np.load('/content/test_stack_output.npy')#, mmap_mode="r+")
test_y = np.load('/content/test_y.npy')#, mmap_mode="r+")


plt.rcParams.update({'legend.fontsize': 10,
                    'axes.labelsize': 14, 
                    'axes.titlesize': 14,
                    'xtick.labelsize': 14,
                    'ytick.labelsize': 14})

def test_eval(model, model_name):

    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]

    print("\nTesting ")
    y_pred = model.predict(test_stack_output)

    print("\nClassification Report: Testing ")
    print(classification_report(np.argmax(test_y, axis = 1), y_pred, target_names = class_names))

    print("\nConfusion Matrix: Testing ")
    conf_matrix = confusion_matrix(np.argmax(test_y, axis = 1), y_pred)
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = class_names, title = model_name + " Testing")

    pred_probas = np.zeros((y_pred.size, y_pred.max()+1))
    pred_probas[np.arange(y_pred.size),y_pred] = 1
    
    print("\nROC Curve: Testing")
    plot_roc(pred_probas, test_y, class_names = class_names, title = model_name + " Testing")

def train_fold(model, model_name, train_X, train_y, val_X, val_y, fold_no, best_accuracy):
    
    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]

    print("\nTraining: fold " + str(fold_no))
    model.fit(train_X, np.argmax(train_y, axis = 1))
    
    print("\nValidation: fold " + str(fold_no))
    y_pred = model.predict(val_X)

    print("\nClassification Report: fold " + str(fold_no))
    print(classification_report(np.argmax(val_y, axis = 1), y_pred, target_names = class_names))

    print("\nConfusion Matrix: fold " + str(fold_no))
    conf_matrix = confusion_matrix(np.argmax(val_y, axis = 1), y_pred)
    fold_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = class_names, title = model_name + " fold " + str(fold_no))

    pred_probas = np.zeros((y_pred.size, y_pred.max()+1))
    pred_probas[np.arange(y_pred.size),y_pred] = 1
    
    print("\nROC Curve: fold" + str(fold_no))
    plot_roc(pred_probas, val_y, class_names = class_names, title = model_name + " fold " + str(fold_no))

    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        os.chdir("/content/drive/MyDrive/saved_models")
        dump(model, model_name + ".joblib")
        os.chdir("/content/")

    return (conf_matrix, fold_accuracy, best_accuracy)


def report(model_class, model_name):

    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]
    kfold = model_class(random_state=77)

    fold_no = 1
    best_accuracy = 0
    total_conf_matrix = np.zeros(shape=(3,3))

    for train_indices, val_indices in kfold.split(exp_stack_output, exp_y):

        train_X = exp_stack_output[train_indices]
        train_y = exp_y[train_indices]
        val_X = exp_stack_output[val_indices]
        val_y = exp_y[val_indices]

        model = model_class(random_state=7, C=0.025)

        (conf_matrix, fold_accuracy, best_accuracy) = train_fold(model, model_name, train_X, train_y, val_X, val_y, fold_no, best_accuracy)
        total_conf_matrix += conf_matrix

        fold_no += 1

        for z in range(10):
          gc.collect()

    print("\n Total Confusion Matrix")
    total_accuracy = plot_confusion_matrix(cm = total_conf_matrix, normalize = False,  target_names = class_names, title = model_name)
    print("Total Accuracy: " + str(total_accuracy)) 

    del model
    for z in range(10):
        gc.collect()

    os.chdir("/content/drive/MyDrive/saved_models")
    best_model = load(model_name + ".joblib")
    os.chdir("/content/")
    test_eval(best_model, model_name)


report(LogisticRegression, "Logistic_Regression_Ensemble_Model")

report(SVC, "Support_Vector_RBF_Ensemble_Model")

report(GaussianNB, "Gaussian_Naive_Bayes_Ensemble_Model")

report(KNeighborsClassifier, "K_Neighbors_Ensemble_Model")

report(RandomForestClassifier, "Random_Forest_Ensemble_Model")

report(DecisionTreeClassifier, "Decision_Tree_Ensemble_Model")

report(BaggingClassifier, "Bagging_Ensemble_Model")

report(GradientBoostingClassifier, "Gradient_Boosting_Ensemble_Model")

report(AdaBoostClassifier, "Ada_Boost_Ensemble_Model")

report(XGBClassifier, "XG_Boost_Ensemble_Model")