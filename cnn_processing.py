# Rafael Pires de Lima
# January 2019
# Transfer Learning application
# This file includes routines to split the data, create bottlenecks, train a new model, predict with new model

import os
import pickle
import random
import shutil

import matplotlib
import matplotlib.image as mpimg
import numpy as np
from keras import applications, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import datetime
from matplotlib import pyplot as plt
from matplotlib import style

matplotlib.use('TkAgg')
style.use("seaborn")

verb = 0  # verbose when training

# folders management:
bottleneck_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'bnecks' + os.sep
model_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'models' + os.sep


def model_app(arch, input_tensor):
    """Loads the appropriate convolutional neural network (CNN) model
      Args:
        arch: String key for model to be loaded.
        input_tensor: Keras tensor to use as image input for the model.
      Returns:
        model: The specified Keras Model instance with ImageNet weights loaded and without the top classification layer.
      """
    # function that loads the appropriate model
    if arch == 'Xception':
        model = applications.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('Xception loaded')
    elif arch == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('VGG16 loaded')
    elif arch == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('VGG19 loaded')
    elif arch == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('ResNet50 loaded')
    elif arch == 'InceptionV3':
        model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('InceptionV3 loaded')
    elif arch == 'InceptionResNetV2':
        model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('InceptionResNetV2 loaded')
    elif arch == 'MobileNet':
        model = applications.MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False,
                                       input_tensor=input_tensor)
        print('MobileNet loaded')
    elif arch == 'DenseNet121':
        model = applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('DenseNet121 loaded')
    elif arch == 'NASNetLarge':
        model = applications.NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('NASNetLarge loaded')
    elif arch == 'MobileNetV2':
        model = applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False,
                                         input_tensor=input_tensor)
        print('MobileNetV2 loaded')
    else:
        print('Invalid model selected')
        model = False

    return model


def save_bottleneck_features(train_data_dir, validation_data_dir, bottleneck_name, img_height, img_width, arch,
                             batch_size=1):
    """Saves the bottlenecks of validation and train data.
      Args:
        train_data_dir: String path to a folder containing subfolders of images (training set).
        validation_data_dir: String path to a folder containing subfolders of images (validation set).
        bottleneck_name: String used as main element of bottlenecks files.
        img_height: Integer, image height.
        img_width: Integer, image width.
        arch: String that defines the CNN model to be used.
        batch_size: batch size
      Returns:
        No returns. Saves bottlenecks using bottleneck_name and bottleneck_dir
      """
    # Saves the bottlenecks of validation and train data.
    # Input is path to train_data_dir and validation_data_dir (directories with the images)
    # bottleneck_name is the name to be used for saving
    # bottleneck_dir is defined outside of this function
    # arch is the architecture to be used
    global bottleneck_dir
    datagen = ImageDataGenerator(rescale=1. / 255)

    # check to see if runs/bottleneck path exists
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

    # build the network
    model = model_app(arch, Input(shape=(img_height, img_width, 3)))

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, generator.n // batch_size, verbose=verb)

    # save a tuple of bottlenecks and the corresponding label
    np.save(open(bottleneck_dir + bottleneck_name + '_train.npy', 'wb'),
            bottleneck_features_train)
    np.save(open(bottleneck_dir + bottleneck_name + '_train_labels.npy', 'wb'),
            generator.classes[0:bottleneck_features_train.shape[0]])

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, generator.n // batch_size, verbose=0)

    # save a tuple of bottlenecks and the corresponding label
    np.save(open(bottleneck_dir + bottleneck_name + '_val.npy', 'wb'),
            bottleneck_features_validation)

    np.save(open(bottleneck_dir + bottleneck_name + '_val_labels.npy', 'wb'),
            generator.classes[0:bottleneck_features_validation.shape[0]])

    # finally, save a "dictionary" as the labels are numeric and eventually we want to know the original string label:
    with open(bottleneck_dir + bottleneck_name + '_dict_l', 'wb') as fp:
        pickle.dump(sorted(os.listdir(train_data_dir)), fp)


def train_top_model(bottleneck_name, model_name, arch, img_height, img_width, epochs, opt, batch_size=16):
    """Trains the new classification layer generating the new classification model dependent on the classes we are using.
      Args:
        bottleneck_name: String used as main element of bottlenecks files.
        model_name: String, name of the model to be saved.
        arch: String that defines the CNN model to be used.
        img_height: Integer, image height.
        img_width: Integer, image width.
        epochs: Integer, the number of epochs (iterations on complete training set) to be performed
        opt: String, optimizer to be used.
        batch_size: batch size
      Returns:
        No returns. Trains and saves the model. Opens a tkinter window with training history
      """

    train_data = np.load(open(bottleneck_dir + bottleneck_name + '_train.npy', 'rb'))
    train_labels = np.load(open(bottleneck_dir + bottleneck_name + '_train_labels.npy', 'rb')).reshape(-1)

    validation_data = np.load(open(bottleneck_dir + bottleneck_name + '_val.npy', 'rb'))
    validation_labels = np.load(open(bottleneck_dir + bottleneck_name + '_val_labels.npy', 'rb')).reshape(-1)

    # check to see if runs/model path exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:]))
    top_model.add(Dropout(0.6))  # dropout helps with overfitting
    top_model.add(Dense(len(np.unique(train_labels)), activation='softmax'))

    if opt == 'RMSprop':
        top_model.compile(optimizer='rmsprop',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    if opt == 'SGD':
        top_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_acc', patience=12, verbose=1),
                 ModelCheckpoint(filepath=model_dir + 'tempbm.h5', monitor='val_acc', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001, verbose=1)]

    history = top_model.fit(train_data, train_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(validation_data, validation_labels),
                            shuffle=True,
                            callbacks=callbacks,
                            verbose=verb)

    # reload best model:
    top_model = load_model(model_dir + 'tempbm.h5')
    score = top_model.evaluate(validation_data, validation_labels, verbose=0)
    print('{:22} {:.2f}'.format('Validation loss:', score[0]))
    print('{:22} {:.2f}'.format('Validation accuracy:', score[1]))
    print('')

    # save the entire model:
    # build the network
    base_model = model_app(arch, Input(shape=(img_height, img_width, 3)))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    if opt == 'RMSprop':
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    if opt == 'SGD':
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    model.save(model_dir + model_name + '.hdf5')
    # also save the dictionary label associated with this file for later testing
    shutil.copy2(bottleneck_dir + bottleneck_name + '_dict_l', model_dir + model_name + '_dict_l')
    # delete temporary model file:
    os.remove(model_dir + 'tempbm.h5')

    print('New classification layer training complete.')

    # plotting the metrics
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    ax[0].plot(range(1, len(history.history['acc']) + 1), history.history['acc'])
    ax[0].plot(range(1, len(history.history['acc']) + 1), history.history['val_acc'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='lower right')

    ax[1].plot(range(1, len(history.history['acc']) + 1), history.history['loss'])
    ax[1].plot(range(1, len(history.history['acc']) + 1), history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper right')

    # set up figure
    fig.set_size_inches(w=5, h=7)

    # plt.show(fig)
    plt.savefig(model_dir + model_name + '.png')


def fine_tune_second_step(train_data_dir, validation_data_dir, model_name, epochs, batch_size):
    """
    Fine tune the new model using SGD (we don't use different batch sizes here so, hopefully, we do not have to
    consider hardware limitations).
      Args:
        train_data_dir: String path to a folder containing subfolders of images (training set).
        validation_data_dir: String path to a folder containing subfolders of images (validation set).
        model_name: String, name of the model to be saved.
        epochs: Integer, the number of epochs (iterations on complete training set) to be performed
      Returns:
        No returns. Trains and saves the model. Opens matplotlib with training history
      """
    datagen = ImageDataGenerator(rescale=1. / 255)

    # load the model:
    model = load_model(model_dir + model_name + '.hdf5')

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    tensorboard = TensorBoard(
        log_dir="logs/{}-{}".format(model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")))
    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                 ModelCheckpoint(filepath=model_dir + 'tempbm.h5', monitor='val_acc', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1),
                 tensorboard]

    # get model input parameters:f
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]

    # set upt the flow from directory for train and validation data:
    generator_train = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    generator_val = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=generator_train.n // batch_size,
                                  epochs=epochs,
                                  validation_data=generator_val,
                                  validation_steps=generator_val.n // batch_size,
                                  shuffle=True,
                                  callbacks=callbacks,
                                  verbose=verb)

    # reload best model:
    top_model = load_model(model_dir + 'tempbm.h5')
    score = top_model.evaluate_generator(generator=generator_val, steps=generator_val.n, verbose=0)
    print('{:22} {:.2f}'.format('Validation loss:', score[0]))
    print('{:22} {:.2f}'.format('Validation accuracy:', score[1]))
    print('')

    # sometimes fine tune might not improve validation accuracy, verify:
    initial_model = load_model(model_dir + model_name + '.hdf5')
    initial_model.compile(optimizer=SGD(lr=1e-4, momentum=0.5),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    score_or = initial_model.evaluate_generator(generator=generator_val, steps=generator_val.n, verbose=0)
    print('{:22} {:.2f}'.format('Original Validation loss:', score_or[0]))
    print('{:22} {:.2f}'.format('Original Validation accuracy:', score_or[1]))
    print('')

    if score_or[1] > score[1]:
        print('Fine tune did not improve model accuracy.')
        # this might be due to batch normalization
        # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
        # https://www.youtube.com/watch?v=nUUqwaxLnWs

    model.save(model_dir + model_name + '_fine_tuned.hdf5')
    # also save the dictionary label associated with this model for later testing
    shutil.copy2(model_dir + os.sep + model_name + '_dict_l', model_dir + model_name + '_fine_tuned_dict_l')
    # delete temporary model file:
    os.remove(model_dir + 'tempbm.h5')

    # plotting the metrics
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    ax[0].plot(range(1, len(history.history['acc']) + 1), history.history['acc'])
    ax[0].plot(range(1, len(history.history['acc']) + 1), history.history['val_acc'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='lower right')

    ax[1].plot(range(1, len(history.history['acc']) + 1), history.history['loss'])
    ax[1].plot(range(1, len(history.history['acc']) + 1), history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper right')

    # set up figure
    fig.set_size_inches(w=5, h=7)
    # plt.show(fig)
    plt.savefig(model_dir + model_name + '_fine_tuned.png')

    print('Fine tune complete.')


if __name__ == '__main__':
    print("Starting...")

    # for model selection parameters
    options_dict = {
        'Xception': (299, 299, 3),
        'VGG16': (224, 224, 3),
        'VGG19': (224, 224, 3),
        'ResNet50': (224, 224, 3),
        'InceptionV3': (299, 299, 3),
        'InceptionResNetV2': (299, 299, 3),
        'MobileNet': (224, 224, 3),
        'MobileNetV2': (224, 224, 3),
        'DenseNet121': (224, 224, 3),
        'NASNetLarge': (331, 331, 3)
    }

    # train and validation data folders
    train_data_dir = '../Data/PP_mc_wb_train'
    validation_data_dir = '../Data/PP_mc_wb_validation'
    test_data_dir = '../Data/PP_mc_wb_test'
    ####################################################
    # choose model architecture with weights coming from ImageNet training:

    models_list = ['MobileNetV2', 'VGG19',
                   'InceptionV3', 'ResNet50']

    # number of epochs for training:
    epochs = 50

    # optimizer
    opt = 'SGD'

    for m in models_list:
        print(m)

        # model and bottleneck names:
        bottleneck_name = m + '_bn'

        # image height and width ("picture size for the model"):
        height = options_dict[m][0]
        width = options_dict[m][1]
        # calling the functions:
        # save the bottlenecks
        save_bottleneck_features(train_data_dir, validation_data_dir, bottleneck_name, height, width, m)
        # then train the top model
        train_top_model(bottleneck_name, m, m, height, width, epochs, opt)
        # fine tune the model:
        if m == 'InceptionV3' or m == 'ResNet50':
            fine_tune_second_step(train_data_dir, validation_data_dir, m, epochs,
                                  batch_size=8)  # 8 for inception/resnet (personal memory limitations)
        else:
            fine_tune_second_step(train_data_dir, validation_data_dir, m, epochs,
                                  batch_size=16)

    # after the models are trained, evaluate the metrics:
    datagen = ImageDataGenerator(rescale=1. / 255)

    print('\n\n')
    for m in models_list:
        print('Evaluating model {}'.format(m))
        # image height and width ("picture size for the model"):
        height = options_dict[m][0]
        width = options_dict[m][1]

        generator_test = datagen.flow_from_directory(
            test_data_dir,
            target_size=(width, height),
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

        # load the model
        this_model = load_model(model_dir + m + '.hdf5')
        this_model.compile(optimizer=SGD(lr=1e-4, momentum=0.5),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        print('----Classification only')
        score = this_model.evaluate_generator(generator=generator_test, steps=generator_test.n, verbose=0)
        print('----{:22} {:.2f}'.format('Validation loss:', score[0]))
        print('----{:22} {:.2f}'.format('Validation accuracy:', score[1]))
        print('')

        # load the fine tuned model
        this_model = load_model(model_dir + m + '_fine_tuned.hdf5')

        print('----Fine tune')
        score = this_model.evaluate_generator(generator=generator_test, steps=generator_test.n, verbose=0)
        print('----{:22} {:.2f}'.format('Validation loss:', score[0]))
        print('----{:22} {:.2f}'.format('Validation accuracy:', score[1]))
        print('')

    print('Complete')
