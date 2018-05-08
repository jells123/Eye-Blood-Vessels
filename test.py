import helpers
from keras.models import load_model
from sklearn.metrics import mean_squared_error

import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random

model = load_model(os.getcwd() + "/models/1-10-0.93.hdf5")

eyes_path = os.getcwd() + "/images" + "/"
vessels_path = os.getcwd() + "/vessels" + "/"
masks_path = os.getcwd() + "/mask" + "/"

sample_size = 49
channels = 3
pad_size = int( np.floor(sample_size/2.0) )

resize_scale = 0.25
test_count = 5

test_imgs = os.listdir(eyes_path)
test_vessels = os.listdir(vessels_path)
masks = os.listdir(masks_path)
random.shuffle(test_imgs)

for t in test_imgs[:6]:
    name = t[:t.lower().find(".jpg")]
    v = helpers.find_in(name, test_vessels)
    m = helpers.find_in(name, masks)

    eye_img = helpers.load_color_img(eyes_path + t)
    vessel_img = helpers.load_gray_img(vessels_path + v)
    mask_img = helpers.load_gray_img(masks_path + m)

    eye_img = cv.resize(eye_img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
    vessel_img = cv.resize(vessel_img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
    mask_img = cv.resize(mask_img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

    vessel_img[vessel_img > 0.0] = 1.0

    eye_padded = helpers.pad_img(eye_img, pad_size)
    vessel_padded = helpers.pad_img(vessel_img, pad_size)

    vessels_predicted_img = np.zeros(vessel_img.shape, dtype=np.float32)
    probs_array = np.zeros(vessel_img.shape, dtype=np.float32)

    samples = [
        [eye_padded[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1]
         for y in range(pad_size, eye_padded.shape[1] - pad_size)]
                for x in range(pad_size, eye_padded.shape[0] - pad_size - 1)
    ]

    print("prediction start...")
    for idx, line in enumerate(samples):
        line = np.array(line, dtype=np.float32)
        predicted_classes = model.predict(line)
        predicted_classes = np.array(list(map(lambda x: np.argmax(x), predicted_classes)), dtype=np.float32)

        predicted_probs = np.array(list(map(lambda x: np.max(x) * np.argmax(x) * -1, predicted_classes)), dtype=np.float32)
        # ??? każdy głosuje na swój i potem średnia z otoczenia, albo każdy głosuje na swoje najbliższe otoczenie?
        for p_idx, p in enumerate(predicted_probs):
            try:
                probs_array[idx-1][p_idx-1 : p_idx+2] += p
                probs_array[idx][p_idx-1 : p_idx+2] += p
                probs_array[idx+1][p_idx-1 : p_idx+2] += p
            except IndexError:
                pass
        vessels_predicted_img[idx] = predicted_classes
        if idx % 250 == 0:
            print("    {} predicted so far...".format(idx))
            cv.imwrite("{}-result.png".format(name), vessels_predicted_img * 255.0)
    print("predicted!")
    cv.imwrite("{}-result.png".format(name), vessels_predicted_img * 255.0)

    probs_array += (np.min(probs_array) * -1.0)
    probs_array /= np.max(probs_array)
    cv.imwrite("{}-sth.png".format(name), probs_array * 255.0)

    for r in range(15):
        x, y = random.randint(0, vessels_predicted_img.shape[0] - 1), random.randint(0, vessels_predicted_img.shape[1] - 1)
        print("predicted: {}, probs: {}, real: {}".format(vessels_predicted_img[x][y], probs_array[x][y], vessel_img[x][y]))

    error = 0.0
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0

    for x in range(0, len(vessels_predicted_img)):
        for y in range(0, len(vessels_predicted_img[x])):
            if mask_img[x][y] == 1.0:
                real = vessel_img[x][y]
                predict = vessels_predicted_img[x][y]
                if real == 1.0 and predict == 1.0:
                    TP += 1.0
                elif real == 0.0 and predict == 1.0:
                    FP += 1.0
                elif real == 1.0 and predict == 0.0:
                    FN += 1.0
                elif real == 0.0 and predict == 0.0:
                    TN += 1.0

    d = vessels_predicted_img.shape[0] * vessels_predicted_img.shape[1]
    print("({}) - ({}, {}, {}, {})".format(d, TP / d, FP / d, TN / d, FN / d))
    error /= d
    print("Error: {}".format(error))

    plt.imshow(vessels_predicted_img, cmap='gray')
    plt.show()