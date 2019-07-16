# -*- coding: utf-8 -*-
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model

DATA_FORMAT = 'channels_last'


def model_builder(input_dim):
    """Builds a model.

    Builds and returns a deep learning model for use in galactic image
    processing applications.

    Parameters
    ----------
    input_dim : tuple of int
        Tuple specifying the dimensions of the image input

    Returns
    -------
    model : :obj:`keras.models.Model`
        The model defined in this file.

    """
    img_input = Input(shape=input_dim)

    cnn = Conv2D(filters=32, kernel_size=6,
                 data_format=DATA_FORMAT, activation='relu')(img_input)
    cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)
    cnn = Conv2D(filters=64, kernel_size=5,
                 data_format=DATA_FORMAT, activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)
    cnn = Conv2D(filters=128, kernel_size=3,
                 data_format=DATA_FORMAT, activation='relu')(cnn)
    cnn = Conv2D(filters=128, kernel_size=3,
                 data_format=DATA_FORMAT, activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(cnn)

    cnn = Flatten(data_format=DATA_FORMAT)(cnn)
    cnn = Dropout(.5)(cnn)

    t01 = Dense(64, activation='relu')(cnn)
    t01 = Dropout(.5)(t01)
    t01_out = Dense(3, activation='softmax', name='t01')(t01)

    # example of a more complicated setup, involving concatenating
    # t01 info + the CNN itself. May be useful for more complicated
    # goals in the future

    # t02 = keras.layers.concatenate([t01, flat])
    # t02 = Dense(2048, activation='relu')(t02)
    # t02_out = Dense(2, activation='softmax', name='t02')(t02)

    model = Model(inputs=img_input, outputs=[t01_out])

    return model
