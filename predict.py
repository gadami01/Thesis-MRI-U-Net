from __future__ import print_function

import os

import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512
img_cols = 512

#137,116,200,209,
imgName = '116.tif'

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

    model.compile(optimizer=Adam(learning_rate=1e-6), loss=dice_coef_loss, metrics=[dice_coef])

    return model

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

print('-' * 30)
print('Loading and preprocessing test data...')
print('-' * 30)
arr = np.empty((1,512,512))
arr[0] = cv2.imread("demo/test/"+ imgName, cv2.IMREAD_GRAYSCALE)
img_test = arr
print(img_test.shape)
imgs_test = normalize(img_test)

imgs_test = np.array(imgs_test).astype('float32')

print('-' * 30)
print('Loading saved weights...')
print('-' * 30)
model = get_unet()
model.load_weights('finalweights.h5')

print('-' * 30)
print('Predicting masks on test data...')
print('-' * 30)
imgs_mask_test = model.predict(imgs_test, verbose=1)
print(imgs_mask_test.shape)

print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)
image = np.array(imgs_mask_test[0] * 255).astype(np.uint8)
cv2.imwrite('demo/preds/pred.png', image, )

originalmask = cv2.imread('demo/doctor/' + imgName[:-4] + '_mask.tif', cv2.COLOR_RGB2GRAY)

plt.figure('Results')

plt.subplot(131), plt.imshow(imgs_test[0], 'gray'), plt.title('Original Image')
plt.axis("off")

plt.subplot(132), plt.imshow(originalmask, 'gray'), plt.title('Original Mask')
plt.axis("off")

plt.subplot(133), plt.imshow(image, 'gray'), plt.title('Prediction Mask')
plt.axis("off")

plt.show()

tp = 0
tn = 0
fp = 0
fn = 0

compare = np.subtract(image[:,:,0], originalmask[:,:,0])
print(image[:,:,0].shape)
print(originalmask[:,:,0].shape)
print(compare.shape)
for i in range(0, 512):
    for j in range(0, 512):
        if compare[i][j] == 0:
            if image[i, j, 0] == 0:
                tn += 1
            if image[i, j, 0] == 255:
                tp += 1
        elif compare[i, j] == 1:
            fp += 1
        else:
            fn += 1

sum = tp + tn + fp + fn
print("\nTrue Positive Pixels : " + str(tp) + " (" + str("{:.2f}".format(tp / sum * 100)) + "%)")
print("True Negative Pixels : " + str(tn) + " (" + str("{:.2f}".format(tn / sum * 100)) + "%)")
print("False Positive Pixels : " + str(fp) + " (" + str("{:.2f}".format(fp / sum * 100)) + "%)")
print("False Negative Pixels : " + str(fn) + " (" + str("{:.2f}".format(fn / sum * 100)) + "%)")

if 2*tp + fp + fn != 0:
    print("Dice Similarity Coefficient : " + str("{:.2f}".format(2*tp/(2*tp + fp + fn))))

