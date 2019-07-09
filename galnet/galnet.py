#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model import model_builder

# top level hyperparameter definitions
INPUT_DIM = (69, 69)
DATA_FORMAT = 'channels_last'
BATCH_SIZE = 256
EPOCHS = 100


def main(argv):
    # safely create output directory for our model/statistics
    # we could also input a unique stamp here, if we want to keep
    # multiple separate (but overall compatible) models
    model_path = os.path.join(argv.DATA, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # import our dataset, which is a pandas dataframe containing path
    # information to the actual image data
    gz2 = pd.read_hdf(os.path.join(argv.DATA, 'gz2.h5'))
    gz2 = gz2[gz2['a01'] >= .3]  # we adopt the agreement threshold of DS18

    # choose the questions we want to classify. note that the number
    # of columns here should match the number of output layers in
    # model.py
    classcols = ['t01']

    # split the data into a training set and a test set. the test set
    # will be set aside entirely until the very end
    train, test = train_test_split(gz2, test_size=.25)

    # if we only want to test, we need to do a lot less
    if argv.TEST:
        model = load_model(os.path.join(model_path, 'model.h5'))

        testgen = ImageDataGenerator()
        testgen = testgen.flow_from_dataframe(test,
                                              directory=MODULE_PATH,
                                              x_col='imgpath',
                                              y_col=classcols,
                                              batchsize=BATCH_SIZE,
                                              target_size=INPUT_DIM,
                                              class_mode='multi_output')

        model.predict_generator(testgen)

    elif argv.TRAIN:
        # create an ImageDataGenerator, which applies random affine
        # transformations to the data. such augmentation is standard
        datagen = ImageDataGenerator(rotation_range=360, zoom_range=[.75, 1.3],
                                     width_shift_range=.05,
                                     height_shift_range=.05,
                                     horizontal_flip=True, vertical_flip=True,
                                     validation_split=.25)

        # create two sets of generators, one for training data and one
        # for validation data, which can be used to check progress
        # throughout training. the target_size option automatically
        # scales our data to the requested size. We also set up for a
        # multi-output model, even though we are currently only
        # checking one question, which will allow some flexibility
        # should this goal change
        traingen = datagen.flow_from_dataframe(train,
                                               directory=MODULE_PATH,
                                               x_col='imgpath',
                                               y_col=classcols,
                                               batchsize=BATCH_SIZE,
                                               target_size=INPUT_DIM,
                                               class_mode='multi_output',
                                               subset='training')

        valgen = datagen.flow_from_dataframe(train,
                                             directory=MODULE_PATH,
                                             x_col='imgpath',
                                             y_col=classcols,
                                             batchsize=BATCH_SIZE,
                                             target_size=INPUT_DIM,
                                             class_mode='multi_output',
                                             subset='validation')

        # now we actually build the model, which is defined in model.py
        model = model_builder(input_dim=traingen.image_shape)

        # compile the model. note that the names of outputs in dicts
        # (e.g., 't01') should match the names of the relevant output
        # layers found in the model definition
        model.compile(optimizer=Adam(lr=0.0001),  # DS18
                      loss={'t01': 'categorical_crossentropy'},
                      loss_weights={'t01': 1.},
                      metrics=['accuracy'])

        # save an image of the model as defined in model.py. can be
        # useful for quickly checking that you have the architecture
        # you want
        plot_model(model, to_file=os.path.join(model_path, 'model.png'),
                   show_shapes=True, show_layer_names=True)

        # calculate the number of steps per epoch (or validation) such
        # that all (or nearly all) images are used
        train_step_size = traingen.n // traingen.batch_size
        val_step_size = valgen.n // valgen.batch_size

        # save the model after each epoch if it's an
        # improvement over previous epochs
        ckpt = ModelCheckpoint(os.path.join(model_path, 'model.h5'),
                               monitor='val_loss', save_best_only=True)
        history = model.fit_generator(generator=traingen,
                                      steps_per_epoch=train_step_size,
                                      validation_data=valgen,
                                      validation_steps=val_step_size,
                                      epochs=EPOCHS,
                                      callbacks=[ckpt],
                                      verbose=1)

        # XXX: the following graphs are only computed for the current
        #      training session. This is ok until we decide to continue
        #      training on a model, instead of starting fresh.

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(model_path, 'acc.pdf'))

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(model_path, 'val.pdf'))

        # save all metrics
        # XXX: will need to append history if we continue training a model
        np.save(os.path.join(model_path, 'acc.npy'),
                history.history['acc'])
        np.save(os.path.join(model_path, 'val_acc.npy'),
                history.history['val_acc'])
        np.save(os.path.join(model_path, 'loss.npy'),
                history.history['loss'])
        np.save(os.path.join(model_path, 'val_loss.npy'),
                history.history['val_loss'])

        # test the model with the test set. it has never seen this data.
        testgen = ImageDataGenerator()
        testgen = testgen.flow_from_dataframe(test,
                                              directory=MODULE_PATH,
                                              x_col='imgpath',
                                              y_col=classcols,
                                              batchsize=BATCH_SIZE,
                                              target_size=INPUT_DIM,
                                              class_mode='multi_output')

        test_step_size = testgen.n // testgen.batch_size

        model.predict_generator(testgen, steps=test_step_size, verbose=1)


if __name__ == '__main__':
    # set up an absolute path to make life easier
    MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..'))

    # set up plotting defaults
    plt.rcParams.update({'font.size': 14,
                         'figure.figsize': (12, 12)})

    # set up command line options for use with hpc
    PARSER = argparse.ArgumentParser(description="Run a CNN.")

    # independent commands
    PARSER.add_argument('-d', '--data', dest='DATA', action='store',
                        default='data', help="Data folder.")

    # run mode. users must select either training or testing
    MODE = PARSER.add_mutually_exclusive_group()
    MODE.add_argument('--train', dest='TRAIN', action='store_true',
                      default=False)
    MODE.add_argument('--test', dest='TEST', action='store_true',
                      default=False)

    args = PARSER.parse_args()
    args.DATA = os.path.join(MODULE_PATH, args.DATA)
    main(args)
