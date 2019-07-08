# -*- coding: utf-8 -*-
import os

from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.utils import plot_model


DATA_FORMAT = 'channels_last'

# configure as needed for your model
LOSS = {'t01': 'categorical_crossentropy'}
LOSS_WEIGHTS = {'t01': 1.}


def model_builder(input_dim, path=''):
    img_input = Input(shape=input_dim)

    cnn = Conv2D(filters=32, kernel_size=6,
                 data_format=DATA_FORMAT, activation='relu')(img_input)
    cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)
    cnn = Conv2D(filters=64, kernel_size=5,
                 data_format=DATA_FORMAT, activation='relu')(img_input)
    cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)
    cnn = Conv2D(filters=128, kernel_size=3,
                 data_format=DATA_FORMAT, activation='relu')(img_input)
    cnn = Conv2D(filters=128, kernel_size=3,
                 data_format=DATA_FORMAT, activation='relu')(img_input)
    cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)

    flat = Flatten(data_format=DATA_FORMAT)(cnn)
    flat = Dropout(.5)(flat)

    t01 = Dense(64, activation='relu')(flat)
    t01 = Dropout(.5)(t01)
    t01_out = Dense(3, activation='softmax', name='t01')(t01)

    # example of a more complicated setup, involving concatenating
    # t01 info + the CNN itself. May be useful for more complicated
    # goals in the future

    # t02 = keras.layers.concatenate([t01, flat])
    # t02 = Dense(2048, activation='relu')(t02)
    # t02_out = Dense(2, activation='softmax', name='t02')(t02)

    model = Model(inputs=img_input, outputs=[t01_out])

    model.compile(optimizer='adam',
                  loss=LOSS,
                  loss_weights=LOSS_WEIGHTS,
                  metrics=['accuracy'])

    # save an image of the model as defined above. can be useful for
    # quickly checking that you have the architecture you want
    plot_model(model, to_file=os.path.join(path, 'model.png'),
               show_shapes=True, show_layer_names=True)

    return model
