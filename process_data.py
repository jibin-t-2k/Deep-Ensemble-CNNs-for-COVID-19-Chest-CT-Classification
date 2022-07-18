import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

from google.colab import drive
drive.mount('/content/drive')

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH_1 = "" # large-covid19-ct-slice-dataset
IMAGE_SIZE = [224, 224]
IMAGE_SHAPE = 224
EPOCHS = 30

filenames = tf.io.gfile.glob(str(GCS_PATH_1 + "/curated_data/curated_data/*/*.png"))
train_filenames, test_filenames = train_test_split(filenames, test_size=0.05,  random_state = 7)

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


train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)

train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


train_X = []
train_y = []
for image, label in train_ds:
    train_X.append(image.numpy().astype("float16"))
    train_y.append(label.numpy())
train_X = np.array(train_X)
train_y = np.array(train_y)

np.save('/content/drive/MyDrive/COVID_CT/train_X.npy', train_X)
np.save('/content/drive/MyDrive/COVID_CT/train_y.npy', train_y)

test_X = []
test_y = []
for image, label in test_ds:
    test_X.append(image.numpy().astype("float16"))
    test_y.append(label.numpy())
test_X = np.array(test_X)
test_y = np.array(test_y)

np.save('/content/drive/MyDrive/COVID_CT/test_X.npy', test_X)
np.save('/content/drive/MyDrive/COVID_CT/test_y.npy', test_y) 