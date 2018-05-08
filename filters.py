from __future__ import division

import os
from os import listdir
from os.path import isfile, join

import cv2
from matplotlib import pylab as plt
from pylab import *
from sklearn.metrics import mean_squared_error

warnings.simplefilter("ignore")


def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)


def confusion_matrix(img, org, mask):
    if img.shape != org.shape:
        org = cv2.resize(org, (img.shape[1], img.shape[0]))

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask / 255.0

    if np.max(img) > 1:
        img = img / 255.0
    if np.max(org) > 1:
        org = org / 255.0

    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i, j] > 0:
                if img[i, j] > 0 and org[i, j] > 0:
                    TP += 1
                elif img[i, j] == 0 and org[i, j] == 0:
                    TN += 1
                elif img[i, j] == 0 and org[i, j] > 0:
                    FP += 1
                elif img[i, j] > 0 and org[i, j] == 0:
                    FN += 1
                else:
                    print("Sth is no yes\n")
                    print(img[i, j])
                    print(org[i, j])
                    return

    print('Macierz pomylek:\n{} TP\t{} FP\n{} FN\t{} TN'.format(TP, FP, FN, TN))
    print('Precision: %5.3f' % (TP / (TP + FP)))
    print('Recall: %5.3f' % (TP / (TP + FN)))
    print('Accuracy: %5.3f\n' % ((TP + TN) / (TP + FP + TN + FN)))


def fun(im, maska):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # imgray = cv2.resize(imgray, dsize=(0, 0), fx=0.25, fy=0.25)

    maska = cv2.cvtColor(maska, cv2.COLOR_BGR2GRAY)
    # maska = cv2.resize(maska, dsize=(0, 0), fx=0.25, fy=0.25)

    maska = maska/255.0
    imgray = np.asarray(np.multiply(imgray, maska), dtype=np.uint8)

    # plt.figure(figsize=(10, 30))
    # imgray = cv2.medianBlur(imgray, 5)

    imgray = cv2.GaussianBlur(imgray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgray = clahe.apply(imgray)

    # imgray = gamma_correction(imgray, 0.9)
    # plt.subplot(5, 1, 1)
    # plt.imshow(imgray, cmap='gray')

    imgray = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

    # plt.subplot(5, 1, 2)
    # plt.imshow(imgray, cmap='gray')

    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # plt.subplot(5, 1, 3)
    # plt.imshow(imgray, cmap='gray')

    imgray = cv2.medianBlur(imgray, 11)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    imgray = cv2.medianBlur(imgray, 11)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    imgray = cv2.medianBlur(imgray, 11)
    # imgray = cv2.fastNlMeansDenoising(imgray)

    # plt.subplot(5, 1, 4)
    # plt.imshow(cv2.bitwise_not(imgray.copy()), cmap='gray')

    black = np.ones(shape=imgray.shape, dtype=np.float32) * 255
    img, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # if cv2.contourArea(cnt) <= 200:
        # if len(cnt) > size:
        cv2.drawContours(black, [cnt], -1, 0, -1)

    # plt.subplot(5, 1, 5)
    # plt.imshow(black, cmap='gray')

    # plt.savefig('try.pdf')

    return cv2.bitwise_not(imgray)


def load():
    images = [f for f in listdir(os.getcwd() + '//all/images/') if isfile(join(os.getcwd() + '//all/images/', f))]
    manual = [f for f in listdir(os.getcwd() + '//all/manual1/') if isfile(join(os.getcwd() + '//all/manual1/', f))]
    masks = [f for f in listdir(os.getcwd() + '//all/mask/') if isfile(join(os.getcwd() + '//all/mask/', f))]

    # plt.figure(figsize=(20, 50))

    for i in range(len(images)):
        im = cv2.imread(os.getcwd() + '//all/images/' + images[i])
        mask = cv2.imread(os.getcwd() + '//all/mask/' + masks[i])

        org = cv2.imread(os.getcwd() + '//all/manual1/' + manual[i])
        org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        org = cv2.resize(org, dsize=(0, 0), fx=0.25, fy=0.25)

        result = fun(im, mask)
        result = cv2.resize(result, dsize=(0, 0), fx=0.25, fy=0.25)

        mask = cv2.resize(mask, dsize=(0, 0), fx=0.25, fy=0.25)

        cv2.imwrite(images[i].split(".")[0] + "-filter-res.png", result)
        # result = extract_bv(im)

        print(images[i].split(".")[0] + "\nRMSE: " + str(mean_squared_error(result, org)))
        confusion_matrix(result, org, mask)

        # plt.subplot(len(images), 2, i*2 + 1)
        # plt.imshow(org, cmap='gray')
        # plt.xticks([]), plt.yticks([])

        # plt.subplot(len(images), 2, i*2 + 2)
        # plt.imshow(result, cmap='gray')
        # plt.xticks([]), plt.yticks([])

    # plt.savefig("result.pdf")


load()
