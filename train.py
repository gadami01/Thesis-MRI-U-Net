from __future__ import print_function

import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
#   import matplotlib.pyplot as plt
import random

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512
img_cols = 512

sett = "Norm_DAZ_BN2_B32_LR5_512_20"


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)

    for i in range(imgs.shape[0]):
        imgs_p[i] = cv2.resize(imgs[i], (img_cols, img_rows))

    return imgs_p


def normalize(imgs):
    # Normalize
    total = imgs.shape[0]

    newimg = np.empty(imgs.shape, dtype=np.uint8)

    for q in range(0, total):
        minimum = imgs[q].min()
        maximum = imgs[q].max()
        newimg[q] = (imgs[q] - minimum) * 255.0 / (maximum - minimum)

        minimum = newimg[q].min()
        maximum = newimg[q].max()
        if minimum != 0 or maximum != 255:
            print(minimum)
            print(maximum)

    print("After Normalize Size :" + str(newimg.shape))

    return newimg


def augmentation(imgs_train, imgs_mask_train):
    print("Original Size :" + str(imgs_train.shape))

    # Rotate Randomly
    total = imgs_train.shape[0]

    newimg = np.empty(imgs_train.shape, dtype=np.uint8)
    newmask = np.empty(imgs_train.shape, dtype=np.uint8)
    for q in range(0, total):
        angle = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((int(img_rows / 2), int(img_cols / 2)), angle, 1)   # Center,angle
        newimg[q] = cv2.warpAffine(imgs_train[q], M, (img_rows, img_cols))
        newmask[q] = cv2.warpAffine(imgs_mask_train[q], M, (img_rows, img_cols))

    imgs_train = np.vstack([imgs_train, newimg])
    imgs_mask_train = np.vstack([imgs_mask_train, newmask])

    print("After Rotation Size :" + str(imgs_train.shape))

    # Flip Horizontal
    total = imgs_train.shape[0]

    newimg = np.empty(imgs_train.shape, dtype=np.uint8)
    newmask = np.empty(imgs_train.shape, dtype=np.uint8)
    for q in range(0, total):
        newimg[q] = np.flip(imgs_train[q], axis=1)
        newmask[q] = np.flip(imgs_mask_train[q], axis=1)

    imgs_train = np.vstack([imgs_train, newimg])
    imgs_mask_train = np.vstack([imgs_mask_train, newmask])

    print("After Flip Size :" + str(imgs_train.shape))

    # Zoom In Randomly
    total = imgs_train.shape[0]

    newimg = np.empty(imgs_train.shape, dtype=np.uint8)
    newmask = np.empty(imgs_train.shape, dtype=np.uint8)
    for q in range(0, total):
        randomCrop = random.randint(int(7*img_rows / 10), int(9*img_rows / 10))     # 70-90% random zoom

        startY = random.randint(0, img_rows - randomCrop)
        startX = random.randint(0, img_cols - randomCrop)

        ni = imgs_train[q, startY:startY + randomCrop, startX:startX + randomCrop]
        nm = imgs_mask_train[q, startY:startY + randomCrop, startX:startX + randomCrop]

        newimg[q] = cv2.resize(ni, (img_rows, img_cols))
        newmask[q] = cv2.resize(nm, (img_rows, img_cols))

    imgs_train = np.vstack([imgs_train, newimg])
    imgs_mask_train = np.vstack([imgs_mask_train, newmask])

    print("After Crop size :" + str(imgs_train.shape))

    return imgs_train, imgs_mask_train


def saveFinalDataset(imgs_train, imgs_mask_train):
    total = imgs_train.shape[0]

    print("Saving finaldataset...")
    for q in range(0, total):
        trainimg = Image.fromarray(np.array(imgs_train[q]).astype(np.uint8), mode="L")
        trainimg = ImageOps.grayscale(trainimg)
        trainimg.save("C:/Users/User/Desktop/Adamides/finaldataset/train" + str(q) + ".tif")

        testimg = Image.fromarray(np.array(imgs_mask_train[q]).astype(np.uint8), mode="L")
        testimg = ImageOps.grayscale(testimg)
        testimg.save("C:/Users/User/Desktop/Adamides/finaldataset/mask" + str(q) + ".tif")


def train_and_predict():
    print('-' * 40)
    print('Loading and preprocessing train data...')
    print('-' * 40)
    imgs_train, imgs_mask_train = load_train_data()

    # imgs_train = preprocess(imgs_train)
    # imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = normalize(imgs_train)

    imgs_train, imgs_mask_train = augmentation(imgs_train, imgs_mask_train)  # Data Augmentation
    sh = np.random.permutation(imgs_train.shape[0])
    imgs_train = imgs_train[sh]
    imgs_mask_train = imgs_mask_train[sh]

    #   saveFinalDataset(imgs_train, imgs_mask_train)                           # Run only once

    imgs_train = np.array(imgs_train).astype('float32')

    imgs_mask_train = np.array(imgs_mask_train).astype('float32')
    imgs_mask_train /= 255  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    tb = TensorBoard(log_dir="logs/" + sett)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, validation_split=0.2, epochs=20, verbose=1, callbacks=[model_checkpoint, tb])

    #model.save("weights.h5")

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()

    #   imgs_test = preprocess(imgs_test)

    imgs_test = normalize(imgs_test)

    imgs_test = np.array(imgs_test).astype('float32')

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'new/' + sett
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = np.array(image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(pred_dir, str(image_id) + '_pred.png'), image, )


if __name__ == '__main__':
    train_and_predict()
