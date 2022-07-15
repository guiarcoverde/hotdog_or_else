import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def train_datagen():
    train_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    return train_gen


def train_set(train_gen, size):
    return train_gen.flow_from_directory('seefood/train', target_size=size, batch_size=32, class_mode='binary')


def test_datagen():
    test_gen = ImageDataGenerator(rescale=1. / 255)
    return test_gen


def testing_set(test_gen, size):
    return test_gen.flow_from_directory('seefood/test', target_size=size, batch_size=32, class_mode='binary')


def cnn_init():
    cnn = tf.keras.models.Sequential()
    return cnn


def first_layer(cnn, filter, kernel, shape):
    cnn.add(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, activation='relu', input_shape=shape))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


def add_layers(cnn, filter, kernel):
    cnn.add(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


def flatten(cnn):
    cnn.add(tf.keras.layers.Flatten())


def full_connect(cnn, qty):
    cnn.add(tf.keras.layers.Dense(units=qty, activation='relu'))


def output_layer(cnn):
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


def compiling(cnn):
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def model_training(cnn):
    history = cnn.fit(training_set, validation_data=test_set, epochs=50)
    return history


def drop(rate):
    cnn.add(tf.keras.layers.Dropout(rate))


def training_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def test_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    train_gen = train_datagen()

    training_set = train_set(train_gen, size=(128, 128))

    test_gen = test_datagen()

    test_set = testing_set(test_gen, size=(128, 128))

    cnn = cnn_init()

    first_layer(cnn, filter=64, kernel=3, shape=[128, 128, 3])

    add_layers(cnn, filter=128, kernel=3)

    add_layers(cnn, filter=128, kernel=3)

    flatten(cnn)

    full_connect(cnn, qty=64)
    drop(rate=0.3)
    full_connect(cnn, qty=64)

    output_layer(cnn)

    cnn.summary()

    compiling(cnn)

    history = model_training(cnn)

    training_plot(history)

    test_plot(history)
