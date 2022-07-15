from keras_preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from tensorflow.keras import optimizers
from keras.layers import Input, AveragePooling2D, Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
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


def model():
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    return resnet_model


def layer_freezing(resnet_model):
    for layer in resnet_model.layers[:15]:
        layer.trainable = False


def transfer_learning(resnet_model):
    x = resnet_model.output
    x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = AveragePooling2D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    transfer_model = Model(inputs=resnet_model.input, outputs=x)
    return transfer_model


def saving():
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
    checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)
    return lr_reduce, checkpoint


def compiling_and_training(transfer_model, lr_reduce, checkpoint):
    learning_rate = 1e-4
    transfer_model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=["accuracy"])
    history = transfer_model.fit(training_set, batch_size=1, epochs=50, validation_data=test_set,
                                 callbacks=[lr_reduce, checkpoint])

    return history

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

    training_set = train_set(train_gen, size=(224, 224))

    test_gen = test_datagen()

    test_set = testing_set(test_gen, size=(224, 224))

    resnet_model = model()

    layer_freezing(resnet_model)

    transfer_model = transfer_learning(resnet_model)

    lr_reduce, checkpoint = saving()

    history = compiling_and_training(transfer_model, lr_reduce, checkpoint)

    training_plot(history)

    test_plot(history)

