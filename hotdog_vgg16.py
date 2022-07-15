import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from tensorflow.keras import layers, models, Model, optimizers
from keras_preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import cv2

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
gpu_options.allow_growth = False
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def vgg():
    vgg_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    return vgg_model


def layer_freezing(vgg_model):
    for layer in vgg_model.layers[:18]:
        layer.trainable = False

    for i, layer in enumerate(vgg_model.layers):
        print(i, layer.name, layer.trainable)


def classi():
    x = vgg_model.output
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    transfer_model = Model(inputs=vgg_model.input, outputs=x)
    return transfer_model


def train_datagen():
    train_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    return train_gen


def train_set(train_gen, size):
    return train_gen.flow_from_directory('seefood/train', target_size=size, batch_size=8, class_mode='binary')


def test_datagen():
    test_gen = ImageDataGenerator(rescale=1. / 255)
    return test_gen


def testing_set(test_gen, size):
    return test_gen.flow_from_directory('seefood/test', target_size=size, batch_size=8, class_mode='binary')


def model_compiling():
    learning_rate = 1e-4
    transfer_model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=["accuracy"])
    history = transfer_model.fit(training_set, epochs=50, validation_data=test_set)

def plotsv2():
    cols = 2
    training_vis = []
    test_vis = []
    fig = plt.figure()
    subfigs = fig.subfigures(ncols=1, nrows=2, squeeze=True, wspace=0, hspace=0)
    subfig1, subfig2 = subfigs
    subfig1.suptitle('Training Set')
    subfig2.suptitle('Test Set')
    label_dict = {0: 'Hot Dog', 1: 'Not Hot Dog'}
    #
    for i in range(cols):
        a = subfig1.add_subplot(cols, 2, i + 1)
        for e in training_set:
            img = e[0][0] * 225
            img = img.astype(np.uint8)
            label = e[1][0]
            if not label in training_vis:
                training_vis.append(label)
            plt.imshow(img)
            a.set_title(label_dict[label])
            break

    for i in range(cols):
        a = subfig2.add_subplot(cols, 2, i + 1)
        for e in test_set:
            img = e[0][0] * 225
            img = img.astype(np.uint8)
            label = e[1][0]
            if not label in test_vis:
                test_vis.append(label)
            plt.imshow(img)
            a.set_title(label_dict[label])
            break

    # # plt.subplots_adjust(ncols=1, nrows=2, squeeze=True, wspace=0, hspace=0)
    plt.show()


if __name__ == '__main__':
    with sess:
        vgg_model = vgg()

        layer_freezing(vgg_model)

        transfer_model = classi()

        train_gen = train_datagen()

        training_set = train_set(train_gen, size=(224, 224))

        test_gen = test_datagen()

        test_set = testing_set(test_gen, size=(224, 224))

        vgg_model.summary()
        #
        model_compiling()