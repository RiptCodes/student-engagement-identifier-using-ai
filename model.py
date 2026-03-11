import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import *



def build_model(n_classes=N_CLASSES, lr=LR):
    inp  = Input(shape=(224, 224, 3))
    base = ResNet50V2(weights='imagenet', include_top=False, input_tensor=inp)

    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    x   = base.output
    x   = GlobalAveragePooling2D()(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.5)(x)
    out = Dense(n_classes, activation='linear', dtype='float32')(x)
    out = tf.keras.layers.Softmax(dtype='float32')(out)

    model = Model(inputs=inp, outputs=out)
    return model
