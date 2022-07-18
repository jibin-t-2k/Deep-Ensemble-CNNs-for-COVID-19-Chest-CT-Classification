import tensorflow as tf

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras import Model, layers, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Average, Maximum, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization


IMAGE_SHAPE = 224


def build_vgg19_model():

    vgg19_base_model = VGG19(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                include_top=False,
                                weights="imagenet")
    vgg19_base_model.trainable = True # 345

    vgg19_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    vgg19_preprocess_input = preprocess_input_vgg19(vgg19_inputs)
    vgg19_output = vgg19_base_model(vgg19_preprocess_input)
    x = Flatten()(vgg19_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    vgg19_model = Model(vgg19_inputs, prediction)

    vgg19_model._name = "VGG19_Model"

    return vgg19_model


def build_resnet101_model():

    resnet101_base_model = ResNet101(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3)
                                    include_top=False,
                                    weights="imagenet")
    resnet101_base_model.trainable = True # 345

    resnet101_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    resnet101_preprocess_input = preprocess_input_resnet101(resnet101_inputs)
    resnet101_output = resnet101_base_model(resnet101_preprocess_input)
    x = Flatten()(resnet101_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    resnet101_model = Model(resnet101_inputs, prediction)

    resnet101_model._name = "Resnet101_Model"
    
    return resnet101_model


def build_inceptionv3_model():

    inceptionv3_base_model = InceptionV3(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                include_top=False,
                                weights="imagenet")
    inceptionv3_base_model.trainable = True # 345

    inceptionv3_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    inceptionv3_preprocess_input = preprocess_input_inceptionv3(inceptionv3_inputs)
    inceptionv3_output = inceptionv3_base_model(inceptionv3_preprocess_input)
    x = Flatten()(inceptionv3_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    inceptionv3_model = Model(inceptionv3_inputs, prediction)

    inceptionv3_model._name = "InceptionV3_Model"

    return inceptionv3_model


def build_resnet152v2_model():

    resnet152v2_base_model = ResNet152V2(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3)
                                    include_top=False,
                                    weights="imagenet")
    resnet152v2_base_model.trainable = True # 345

    resnet152v2_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    resnet152v2_preprocess_input = preprocess_input_resnet152v2(resnet152v2_inputs)
    resnet152v2_output = resnet152v2_base_model(resnet152v2_preprocess_input)
    x = Flatten()(resnet152v2_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    resnet152v2_model = Model(resnet152v2_inputs, prediction)

    resnet152v2_model._name = "ResNet152V2_Model"
    
    return resnet152v2_model


def build_inception_resnetv2_model():  
  
    inception_resnetv2_base_model = InceptionResNetV2(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                            include_top=False,
                            weights="imagenet")
    inception_resnetv2_base_model.trainable = True  #132

    inception_resnetv2_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    inception_resnetv2_preprocess_input = preprocess_input_inception_resnetv2(inception_resnetv2_inputs)
    inception_resnetv2_output = inception_resnetv2_base_model(inception_resnetv2_preprocess_input)
    x = Flatten()(inception_resnetv2_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    inception_resnetv2_model = Model(inception_resnetv2_inputs, prediction)

    inception_resnetv2_model._name = "Inception_ResnetV2_Model"

    return inception_resnetv2_model



def build_densenet201_model():

    densenet201_base_model = DenseNet201(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                                    include_top=False,
                                    weights="imagenet")
    densenet201_base_model.trainable = True  #707

    densenet201_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    densenet201_preprocess_input = preprocess_input_densenet201(densenet201_inputs)
    densenet201_output = densenet201_base_model(densenet201_preprocess_input)
    x = Flatten()(densenet201_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    densenet201_model = Model(densenet201_inputs, prediction)

    densenet201_model._name = "DenseNet201_Model"

    return densenet201_model



def build_xception_model():  
  
    xception_base_model = Xception(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                            include_top=False,
                            weights="imagenet")
    xception_base_model.trainable = True  #132

    xception_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    xception_preprocess_input = preprocess_input_xception(xception_inputs)
    xception_output = xception_base_model(xception_preprocess_input)
    x = Flatten()(xception_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    xception_model = Model(xception_inputs, prediction)

    xception_model._name = "Xception_Model"
    
    return xception_model



def build_efficientb7_model():

    efficientb7_base_model = EfficientNetB7(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                    include_top=False, 
                                    weights="imagenet")
    efficientb7_base_model.trainable = True  #813

    efficientb7_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    efficientb7_preprocess_input = preprocess_input_efficientb7(efficientb7_inputs)
    efficientb7_output = efficientb7_base_model(efficientb7_preprocess_input)
    x = Flatten()(efficientb7_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    efficientb7_model = Model(efficientb7_inputs, prediction)

    efficientb7_model._name = "EfficientNetB7_Model"
    
    return efficientb7_model



def build_mobilenet_v2_model():  
  
    mobilenet_v2_base_model = MobileNetV2(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                            include_top=False,
                            weights="imagenet")
    mobilenet_v2_base_model.trainable = True  #132

    mobilenet_v2_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    mobilenet_v2_preprocess_input = preprocess_input_mobilenet_v2(mobilenet_v2_inputs)
    mobilenet_v2_output = mobilenet_v2_base_model(mobilenet_v2_preprocess_input)
    x = Flatten()(mobilenet_v2_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(3,activation="softmax")(x)

    mobilenet_v2_model = Model(mobilenet_v2_inputs, prediction)

    mobilenet_v2_model._name = "MobileNetV2_Model"

    return mobilenet_v2_model