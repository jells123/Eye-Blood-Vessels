import os
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
from keras import Model, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, core, Convolution2D, merge, MaxPool2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

import helpers


eyes_path = os.getcwd() + "/images" + "/"
vessels_path = os.getcwd() + "/vessels" + "/"
masks_path = os.getcwd() + "/mask" + "/"

samples_count = 200000
no_class_percentage = 0.65
valid_size_percentage = 0.15

sample_size = 49
channels = 1

resize_scale = 0.25

pad_size = int( np.floor(sample_size/2.0) )

def convert_to_training_data(train_x, train_y):
    train_x = train_x.reshape(-1, sample_size, sample_size, 1)
    train_y_onehot = to_categorical(train_y)
    train_x, valid_x, train_label, valid_label = train_test_split(train_x, train_y_onehot,
                                                                  test_size=valid_size_percentage,
                                                                  random_state=13,
                                                                  shuffle=True)
    print(train_x.shape, valid_x.shape, train_label.shape)
    return train_x, valid_x, train_label, valid_label

def gather_data():
    eyes = os.listdir(eyes_path)
    vessels = os.listdir(vessels_path)
    masks = os.listdir(masks_path)

    yes_per_example = int ( samples_count / len(eyes) * (1.0 - no_class_percentage) )
    no_per_example = int ( samples_count / len(eyes) * no_class_percentage )
    N = (yes_per_example+no_per_example) * len(eyes)
#    train_x = np.empty((N, sample_size, sample_size, channels), dtype=np.float32)
    train_x = np.empty((N, sample_size, sample_size), dtype=np.float32)
#    train_x = np.empty((N, channels, sample_size, sample_size), dtype=np.float32)
    train_y = np.empty(N, dtype=np.uint8)

    idx = 0
    print("Loading data...")
    for e in eyes:
        print("-> {}".format(e))
        name = e[:e.lower().find(".jpg")]
        v = helpers.find_in(name, vessels)
        m = helpers.find_in(name, masks)

        eye_img = helpers.load_gray_img(eyes_path + e)
        vessel_img = helpers.load_gray_img(vessels_path + v)
        mask_img = helpers.load_gray_img(masks_path + m)

        eye_img = cv.resize(eye_img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
        vessel_img = cv.resize(vessel_img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
        mask_img = cv.resize(mask_img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

        vessel_img[vessel_img > 0.0] = 1.0
        mask_img[mask_img > 0.0] = 1.0
        #for binary imgs

        eye_pad = helpers.pad_img(eye_img, pad_size)
        vessel_pad = helpers.pad_img(vessel_img, pad_size)

        yes, no = 0, 0

        while yes < yes_per_example or no < no_per_example:
            x, y = random.randint(0, eye_img.shape[0] - 1), random.randint(0, eye_img.shape[1] - 1)
            if mask_img[x][y] != 0.0:
                x += pad_size
                y += pad_size
                sample = eye_pad[x-pad_size:x+pad_size+1, y-pad_size:y+pad_size+1]
#                sample = sample.reshape(-1, sample_size, sample_size, 1)
#                sample = eye_pad[x-pad_size:x+pad_size, y-pad_size:y+pad_size, :]
                train_x[idx] = sample
                if vessel_pad[x][y] == 0.0 and no < no_per_example:
                    train_y[idx] = 0 # NO class
                    no += 1
                    idx += 1
                elif vessel_pad[x][y] != 0.0 and yes < yes_per_example:
                    train_y[idx] = 1 # YES class
                    yes += 1
                    idx += 1

    return train_x, train_y

def get_model_1(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(48, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(360, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(720, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

    return model

def get_model_2(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(80, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(360, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(720, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

def get_model_unet(input_shape, num_classes):
    inputs = Input(shape=(1, sample_size, sample_size))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = Dropout(0.5)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)

    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

def get_model_kaggle(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    return model

train_x, train_y = gather_data()
train_x, valid_x, train_label, valid_label = convert_to_training_data(train_x, train_y)
model = get_model_kaggle(train_x[0].shape, num_classes=2)
filepath = os.getcwd() + "/models/" + "0.25gray-200k+128-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
model.summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
train = model.fit(train_x, train_label, batch_size=32, epochs=50,
                  verbose=1, callbacks=[checkpoint, annealer],
                  validation_data=(valid_x, valid_label),
                  shuffle=True
                 )
