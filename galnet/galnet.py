#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from model import model_builder

INPUT_DIM = (69, 69)
DATA_FORMAT = 'channels_last'
BATCH_SIZE = 256


def main():
    gz2 = pd.read_hdf('../data/gz2.h5')

    train, test = train_test_split(gz2, test_size=.25)

    datagen = ImageDataGenerator(rotation_range=360, zoom_range=[.75, 1.3],
                                 width_shift_range=.05, height_shift_range=.05,
                                 horizontal_flip=True, vertical_flip=True,
                                 validation_split=.25)

    classcols = ['t01']
    traingen = datagen.flow_from_dataframe(train,
                                           directory='..',
                                           x_col='imgpath',
                                           y_col=classcols,
                                           batchsize=BATCH_SIZE,
                                           target_size=INPUT_DIM,
                                           class_mode='multi_output',
                                           subset='training')

    valgen = datagen.flow_from_dataframe(train,
                                         directory='..',
                                         x_col='imgpath',
                                         y_col=classcols,
                                         batchsize=BATCH_SIZE,
                                         target_size=INPUT_DIM,
                                         class_mode='multi_output',
                                         subset='validation')

    model = model_builder(input_dim=traingen.image_shape)

    TRAIN_STEP_SIZE = traingen.n // traingen.batch_size
    VAL_STEP_SIZE = valgen.n // valgen.batch_size
    print(TRAIN_STEP_SIZE)
    print(VAL_STEP_SIZE)

    ckpt = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

    model.fit_generator(generator=traingen,
                        steps_per_epoch=TRAIN_STEP_SIZE,
                        validation_data=valgen,
                        validation_steps=VAL_STEP_SIZE,
                        epochs=10,
                        callbacks=[ckpt],
                        verbose=1)


if __name__ == '__main__':
    main()
