from __future__ import print_function

import os
import cv2
import numpy as np

data_path = "C:/Users/hriro/Desktop/MRIsplitted2/"

image_rows = 256
image_cols = 256


def load_train_data():
    train_data_path = data_path + 'train'
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Loading training images...')
    print('-' * 30)

    for image_name in images:

        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'

        img = cv2.imread(train_data_path + "/" + image_name, cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(train_data_path + "/" + image_mask_name, cv2.IMREAD_GRAYSCALE)

        img = np.array(img)
        img_mask = np.array(img_mask)

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    return imgs, imgs_mask


def load_test_data():
    test_data_path = data_path + 'test'
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('-' * 30)
    print('Loading test images...')
    print('-' * 30)

    for image_name in images:
        img_id = int(image_name.split('.')[0])

        img = cv2.imread(test_data_path + '/' + image_name, cv2.IMREAD_GRAYSCALE)

        img = np.array(img)

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    return imgs, imgs_id
