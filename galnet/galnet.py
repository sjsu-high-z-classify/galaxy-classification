#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

INPUT_DIM = (69, 69)
DATA_FORMAT = 'channels_last'
BATCH_SIZE = 256

gz2 = pd.read_hdf('data/gz2.h5')

train, test = train_test_split(gz2, test_size=.25)

datagen = ImageDataGenerator(rotation_range=360, zoom_range=[.75, 1.3],
                             width_shift_range=.05, height_shift_range=.05,
                             horizontal_flip=True, vertical_flip=True,
                             validation_split=.25
                            )

classcols = ['t01']
agreementcols = ['a01']
loss = {'t01': 'categorical_crossentropy'}
loss_weights = {'t01': 1.}

traingen = datagen.flow_from_dataframe(train, x_col='imgpath', y_col=classcols, batchsize=BATCH_SIZE,
                                       target_size=INPUT_DIM, class_mode='multi_output',
                                       subset='training', sample_weight=agreementcols
                                      )

valgen = datagen.flow_from_dataframe(train, x_col='imgpath', y_col=classcols, batchsize=BATCH_SIZE,
                                     target_size=INPUT_DIM, class_mode='multi_output',
                                     subset='validation', sample_weight=agreementcols
                                    )

img_input = Input(shape=traingen.image_shape)

cnn = Conv2D(filters=32, kernel_size=6, data_format=DATA_FORMAT, activation='relu')(img_input)
cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)
cnn = Conv2D(filters=64, kernel_size=5, data_format=DATA_FORMAT, activation='relu')(img_input)
cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)
cnn = Conv2D(filters=128, kernel_size=3, data_format=DATA_FORMAT, activation='relu')(img_input)
cnn = Conv2D(filters=128, kernel_size=3, data_format=DATA_FORMAT, activation='relu')(img_input)
cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)

flat = Flatten(data_format=DATA_FORMAT)(cnn)
flat = Dropout(.5)(flat)

t01 = Dense(64, activation='relu')(flat)
t01 = Dropout(.5)(t01)
# t01 = Dense(2048, activation='relu')(t01)
# t01 = Dropout(.5)(t01)
t01_out = Dense(3, activation='softmax', name='t01')(t01)

# t02 = keras.layers.concatenate([t01, flat])
# t02 = Dense(2048, activation='relu')(t02)
# t02_out = Dense(2, activation='softmax', name='t02')(t02)

# t03 = keras.layers.concatenate([t02, flat])
# t03 = Dense(2048, activation='relu')(t03)
# t03_out = Dense(2, activation='softmax', name='t03')(t03)

model = Model(inputs=img_input, outputs=[t01_out])

model.compile(optimizer='adam',
              loss=loss,
              loss_weights=loss_weights,
              metrics=['accuracy']
             )

TRAIN_STEP_SIZE = traingen.n // traingen.batch_size
VAL_STEP_SIZE = valgen.n // valgen.batch_size

ckpt = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

model.fit_generator(generator=traingen,
                    steps_per_epoch=TRAIN_STEP_SIZE,
                    validation_data=valgen,
                    validation_steps=VAL_STEP_SIZE,
                    epochs=10,
                    callbacks=[ckpt],
                    verbose=1
                   )