from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model
try:
    from keras_implementation import generator
except:
    import generator

import keras.backend as K
import tensorflow as tf

from skimage.segmentation import mark_boundaries

from PIL import Image

import numpy as np
import os
import shutil

from skimage.segmentation import find_boundaries
from skimage.morphology import dilation


def find_all_samples(path):
    all_samples = os.listdir(path)
    return all_samples


def create_mask(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=int)
        for mask in masks:
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _mask = _mask.resize((width, height))
                _mask = np.array(_mask)
                complete_mask = np.maximum(complete_mask, _mask)
        os.mkdir(os.path.join(sample_path, 'mask'))
        mask_image = Image.fromarray(complete_mask.astype('uint8'), 'L')
        mask_image.save(os.path.join(sample_path, 'mask', '{}.png'.format(sample)))


def create_border_mask(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=int)
        for i, mask in enumerate(masks):
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _array = np.array(_mask.resize((width, height)))
                _array = np.array(_array > 0, dtype=int) * (i + 1)
                complete_mask = np.add(complete_mask, _array)
        full_bounds = np.array(complete_mask, dtype=int)
        bound_array = find_boundaries(label_img = full_bounds, connectivity = 1, mode='outer', background=0)
        bound_array = np.array(bound_array, dtype=int)
        bound_array = dilation(bound_array)
        bound_array = bound_array * 255
        print(np.max(bound_array))

        # save image
        shutil.rmtree(os.path.join(sample_path, 'border'))
        os.mkdir(os.path.join(sample_path, 'border'))
        mask_image = Image.fromarray(bound_array.astype('uint8'), 'L')
        mask_image.save(os.path.join(sample_path, 'border', '{}.png'.format(sample)))


def mean_iou(y_true, y_pred):
    y_true = K.round(y_true)
    print(y_true)
    y_pred = K.round(y_pred)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
    return score


def create_model(filter_size = 8, drop_rate=.4):
    img_input = Input(shape=(256,256,1))

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(2, 2)(conv2)

    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(conv4)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(2, 2)(drop4)

    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(conv5)
    drop5 = Dropout(drop_rate)(conv5)

    # Upconvolutional layers
    uconv4 = Conv2DTranspose(filters=filter_size * 8, kernel_size=2, strides=2, activation='relu', padding='same')(drop5)
    uconc4 = concatenate([drop4, uconv4], axis=3)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconc4)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconv4)

    uconv3 = Conv2DTranspose(filters=filter_size * 4, kernel_size=2, strides=2, activation='relu', padding='same')(uconv4)
    uconc3 = concatenate([conv3, uconv3], axis=3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconc3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconv3)

    uconv2 = Conv2DTranspose(filters=filter_size * 2, kernel_size=2, strides=2, activation='relu', padding='same')(uconv3)
    uconc2 = concatenate([conv2, uconv2], axis=3)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconc2)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv2)

    uconv1 = Conv2DTranspose(filters=filter_size, kernel_size=2, strides=2, activation='relu', padding='same')(uconv2)
    uconc1 = concatenate([conv1, uconv1], axis=3)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(uconc1)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(filters=2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1)

    pred = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(uconv1)

    model = Model(inputs=img_input, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', mean_iou])
    return model



if __name__ == '__main__':
    # path_img = 'C:/Users/huubh/Documents/DSB2018_bak/img_no_masks'
    path_img = 'img'
    model_x2 = create_model()
    model_x2.summary()
    labels = os.listdir(path_img)
    training = labels[:608]
    validation = labels[608:]
    print(len(training))
    print(len(validation))
    training_generator = generator.DataGenerator(training, path_img,
                                                 rotation=True, flipping=True, zoom=1.5, batch_size = 16, dim=(256,256))
    validation_generator = generator.DataGenerator(validation, path_img,
                                                 rotation=True, flipping=True, zoom=False, batch_size = 31, dim=(256,256))
    model_x2.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=64)
    # Save model
    model_x2.save('model_b5.h5')
