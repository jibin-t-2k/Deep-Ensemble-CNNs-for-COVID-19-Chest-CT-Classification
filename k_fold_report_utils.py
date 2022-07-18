import tensorflow as tf
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef

from plot_utils import plot_confusion_matrix, plot_roc, train_curves
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef

import gc

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
    return tf.image.resize(img, [224, 224])

def process_path(file_path):
    gt = get_gt(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, gt

def prepare_for_training(ds, cache=True, shuffle = True, batch_size = 32, autotune=tf.data.experimental.AUTOTUNE, shuffle_buffer_size=1000):

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size = shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    ds = ds.prefetch(buffer_size = autotune)

    return ds


def test_eval(model, test_filenames, autotune=tf.data.experimental.AUTOTUNE, batch_size=32):

    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]

    test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
    TEST_IMG_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
    test_ds = test_list_ds.map(process_path, num_parallel_calls=autotune)
    test_ds = prepare_for_training(test_ds, shuffle=False, batch_size = batch_size, autotune=autotune)

    test_gtds = test_list_ds.map(get_gt, num_parallel_calls=autotune)

    test_gts = []
    for gt in test_gtds:
        test_gts.append(gt.numpy())

    gt_labels = np.argmax(test_gts, axis = 1)
    test_gts = np.array(test_gts)

    print("\nTesting ")
    pred_probas = model.predict(test_ds, steps = (TEST_IMG_COUNT // (batch_size // 1.5)), verbose = 1)
    pred_labels = np.argmax(pred_probas, axis = 1)

    print("\nClassification Report: Testing ")
    print(classification_report(gt_labels, pred_labels[:TEST_IMG_COUNT], target_names = class_names))

    print("\nConfusion Matrix: Testing ")
    conf_matrix = confusion_matrix(gt_labels, pred_labels[:TEST_IMG_COUNT])
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = class_names, title = model._name + " Testing")

    print("\nROC Curve: Testing")
    plot_roc(pred_probas[:TEST_IMG_COUNT], test_gts, class_names = class_names, title = model._name + " Testing")

    

def total_eval(model, filenames, autotune=tf.data.experimental.AUTOTUNE, batch_size=32):

    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]

    test_list_ds = tf.data.Dataset.from_tensor_slices(filenames)
    TEST_IMG_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
    test_ds = test_list_ds.map(process_path, num_parallel_calls=autotune)
    test_ds = prepare_for_training(test_ds, shuffle=False, batch_size = batch_size, autotune=autotune)

    test_gtds = test_list_ds.map(get_gt, num_parallel_calls=autotune)

    test_gts = []
    for gt in test_gtds:
        test_gts.append(gt.numpy())

    gt_labels = np.argmax(test_gts, axis = 1)
    test_gts = np.array(test_gts)

    print("\nTesting ")
    pred_probas = model.predict(test_ds, steps = (TEST_IMG_COUNT // (batch_size // 1.5)), verbose = 1)
    pred_labels = np.argmax(pred_probas, axis = 1)

    print("\nClassification Report: Testing ")
    print(classification_report(gt_labels, pred_labels[:TEST_IMG_COUNT], target_names = class_names))

    print("\nConfusion Matrix: Testing ")
    conf_matrix = confusion_matrix(gt_labels, pred_labels[:TEST_IMG_COUNT])
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = class_names, title = model._name)

    print("\nROC Curve: Testing")
    plot_roc(pred_probas[:TEST_IMG_COUNT], test_gts, class_names = class_names, title = model._name)



def train_fold(model, train_filenames, val_filenames, fold_no, best_accuracy, epochs, autotune=tf.data.experimental.AUTOTUNE, batch_size=32):
    
    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]

    train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames).shuffle(len(train_filenames))
    val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

    TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
    VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()

    train_ds = train_list_ds.map(process_path, num_parallel_calls=autotune)
    val_ds = val_list_ds.map(process_path, num_parallel_calls=autotune)

    train_ds = prepare_for_training(train_ds, shuffle=True, batch_size = batch_size, autotune=autotune)
    val_ds = prepare_for_training(val_ds, shuffle=False, batch_size = batch_size, autotune=autotune)

    val_gtds = val_list_ds.map(get_gt, num_parallel_calls=autotune)

    val_gts = []
    for gt in val_gtds:
        val_gts.append(gt.numpy())

    gt_labels = np.argmax(val_gts, axis = 1)
    val_gts = np.array(val_gts)

    print("\nTraining: fold " + str(fold_no))
    history = model.fit(train_ds,
                        epochs = epochs,
                        steps_per_epoch = TRAIN_IMG_COUNT // batch_size,
                        validation_data = val_ds,
                        validation_steps = VAL_IMG_COUNT // batch_size,
                        verbose = 1,
                        callbacks = [ModelCheckpoint(model._name + "_fold" + str(fold_no) + ".h5",
                                                    verbose = 1,
                                                    monitor = "val_accuracy",
                                                    mod = "max",
                                                    save_best_only = True)])
    
    os.chdir("/content/drive/MyDrive/saved_models/")
    print("\nTraining Curve: fold " + str(fold_no))
    np.save(model._name + "_fold" + str(fold_no) + ".npy", history.history)
    training_history = np.load(model._name + "_fold" + str(fold_no) + ".npy", allow_pickle="TRUE").item()
    train_curves(training_history, model._name + " fold " + str(fold_no))
    os.chdir("/content/")
    
    best_model = load_model(model._name + "_fold" + str(fold_no) + ".h5")

    print("\nValidation: fold " + str(fold_no))
    pred_probas = best_model.predict(val_ds, steps = (VAL_IMG_COUNT // (batch_size // 1.25)), verbose = 1)
    pred_labels = np.argmax(pred_probas, axis = 1)

    print("\nClassification Report: fold " + str(fold_no))
    print(classification_report(gt_labels, pred_labels[:VAL_IMG_COUNT], target_names = class_names))

    print("\nConfusion Matrix: fold " + str(fold_no))
    conf_matrix = confusion_matrix(gt_labels, pred_labels[:VAL_IMG_COUNT])
    fold_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = class_names, title = model._name + " fold " + str(fold_no))
    
    print("\nROC Curve: fold" + str(fold_no))
    plot_roc(pred_probas[:VAL_IMG_COUNT], val_gts, class_names = class_names, title = model._name + " fold " + str(fold_no))

    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        os.chdir("/content/drive/MyDrive/saved_models")
        best_model.save(model._name + ".h5")
        # best_model.save_weights(model_name + "_weights.h5")
        os.chdir("/content/")

    del best_model
    for z in range(10):
      gc.collect()

    return (model._name, conf_matrix, fold_accuracy, best_accuracy)



def report(model, init_weights, exp_filenames, test_filenames, epochs, autotune=tf.data.experimental.AUTOTUNE, batch_size=32):

    class_names = ["Non-COVID", "COVID", "Viral Pneumonia"]
    kfold = KFold(n_splits=5, shuffle=True, random_state=77)

    fold_no = 1
    best_accuracy = 0
    total_conf_matrix = np.zeros(shape=(3,3))

    print("\nModel Summary")
    print(model.summary())

    for train_indices, val_indices in kfold.split(exp_filenames):

        train_filenames = [exp_filenames[i] for i in train_indices]
        val_filenames = [exp_filenames[j] for j in val_indices]

        model.set_weights(init_weights)

        (model_name, conf_matrix, fold_accuracy, best_accuracy) = train_fold(model, train_filenames, val_filenames, fold_no, best_accuracy, epochs, autotune=autotune, batch_size=batch_size)
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
    best_model = load_model(model_name + ".h5")
    os.chdir("/content/")
    test_eval(best_model, test_filenames, epochs, autotune=autotune, batch_size=batch_size)