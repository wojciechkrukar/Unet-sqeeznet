#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.models import Model
from keras.layers import Conv2D, Activation, Reshape, Dropout, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization, Activation
from keras import backend as K


def fire_module(x, fire_id, squeeze=16, expand=64, fire_blockname=''):
    f_name = "fire{0}/{1}"
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(squeeze, (1, 1),
               padding='same',
               name=f_name.format(fire_id, "squeeze1x1"))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    left = Conv2D(expand, (1, 1),
                  padding='same',
                  activation='relu',
                  name=f_name.format(fire_id, "expand1x1"))(x)

    right = Conv2D(expand, (3, 3),
                   padding='same',
                   activation='relu',
                   name=f_name.format(fire_id, "expand3x3"))(x)

    out = concatenate([left, right],
                      axis=channel_axis,
                      name=f_name.format(fire_id, "concat") + fire_blockname)
    return out


def fire_module_up(x1, x2, kernel_size=1, strides=(1, 1), fire_id=0, squeeze=16, expand=64):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x1 = Conv2DTranspose(squeeze, kernel_size,
                         strides=strides,
                         padding='same')(x1)
    up = concatenate([x1, x2], axis=channel_axis)
    up = fire_module(up, fire_id=fire_id,
                     squeeze=squeeze,
                     expand=expand)
    return up


def SqueezeUNet(inputs, num_classes=None, include_top=True, dropout=0.0, fireplus=True, reshape_out=False, activation='sigmoid'):
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation
    :param inputs: input layer.
    :param num_classes: number of classes.
    :param include_top: if true, includes classification layers.
    :param dropout: dropout rate.
    :param fireplus: added two extra fire module.
    :param reshape_out: outputs are reshaped in (row * col, `num_classes`)
    :param activation: type of activation at the top layer.
    :returns: SqueezeUNet model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    if num_classes is None:
        num_classes = input_shape[channel_axis]

    conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv1')(inputs)
    conv1 = BatchNormalization(axis=channel_axis)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

    block1 = fire_module(pool1, fire_id=2, squeeze=16, expand=64)
    block1 = fire_module(block1, fire_id=3, squeeze=16, expand=64, fire_blockname='/block1')
    pool_block1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool_block1')(block1)

    block2 = fire_module(pool_block1, fire_id=4, squeeze=32, expand=128)
    block2 = fire_module(block2, fire_id=5, squeeze=32, expand=128, fire_blockname='/block2')
    pool_block2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool_block2')(block2)

    block3 = fire_module(pool_block2, fire_id=6, squeeze=48, expand=192)
    block3 = fire_module(block3, fire_id=7, squeeze=48, expand=192, fire_blockname='/block3')

    block4 = fire_module(block3, fire_id=8, squeeze=64, expand=256)
    block4 = fire_module(block4, fire_id=9, squeeze=64, expand=256, fire_blockname='/block4')

    # extra fire modules
    if fireplus:
        block5 = fire_module(block4, fire_id=10, squeeze=96, expand=384)
        block5 = fire_module(block5, fire_id=11, squeeze=96, expand=384, fire_blockname='/block5')

    last_block = block4 if not fireplus else block5

    if dropout != 0.0:
        last_block = Dropout(dropout)(last_block)

    up1 = last_block
    if fireplus:
        up1 = fire_module_up(last_block, block4, kernel_size=1,
                                strides=1, fire_id=12, squeeze=64, expand=256)

    up2 = fire_module_up(up1, block3, kernel_size=1,
                            strides=1, fire_id=13,
                            squeeze=48, expand=192)
    up2 = UpSampling2D(size=(2, 2))(up2)

    up3 = fire_module_up(up2, block2, kernel_size=1,
                            strides=1, fire_id=14,
                            squeeze=32, expand=128)
    up3 = UpSampling2D(size=(2, 2))(up3)

    up4 = fire_module_up(up3, block1, kernel_size=1,
                            strides=1, fire_id=15,
                            squeeze=16, expand=64)
    up4 = UpSampling2D(size=(2, 2))(up4)

    up5 = fire_module_up(up4, conv1, kernel_size=1,
                            strides=1, fire_id=16,
                            squeeze=16, expand=32)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(up5)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)

    if include_top:
        x = Conv2D(num_classes, (1, 1), padding='same')(x)

        if reshape_out:
            if K.image_data_format() == 'channels_first':
                channel, row, col = input_shape[1:]
            else:
                row, col, channel = input_shape[1:]

            x = Reshape((row * col, num_classes))(x)

        x = Activation(activation)(x)

    return Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    from keras.layers import Input
    import time
    import numpy as np

    img_rows = 128
    img_cols = 128
    channels = 3
    calculates_inf_time = True

    inputs = Input((img_rows, img_cols, channels))
    model = SqueezeUNet(inputs, fireplus=True, dropout=0.0)

    ip = np.ones((1, img_rows, img_cols, channels), dtype=np.float32)
    res = model.predict(ip, verbose=0)

    ip = np.ones((1, img_rows, img_cols, channels), dtype=np.float32)
    start_t = time.time()
    res = model.predict(ip, verbose=0)
    print("inference time: ", time.time() - start_t)

