import numpy as np
import cv2 as cv

def find_in(name, sth):
    for v in sth:
        if v.startswith(name):
            return v
    print(sth)
    print("Oooops, {} wasn't found".format(name))
    print("(" + sth[0] + ")")
    return -1

def pad_img(img, padsize):
    # rows
    if len(img.shape) == 3:
        upper_pad = img[img.shape[0] - padsize:img.shape[0], :, :]
        lower_pad = img[0:padsize + 1, :, :]
    elif len(img.shape) == 2:
        upper_pad = img[img.shape[0] - padsize:img.shape[0], :]
        lower_pad = img[0:padsize + 1, :]
    img = np.concatenate((upper_pad, img), axis=0)
    img = np.concatenate((img, lower_pad), axis=0)

    # columns
    if len(img.shape) == 3:
        p = np.zeros((img.shape[0], padsize, img.shape[2]), dtype=np.float32)
    elif len(img.shape) == 2:
        p = np.zeros((img.shape[0], padsize), dtype=np.float32)

    img = np.concatenate((p, img), axis=1)
    img = np.concatenate((img, p), axis=1)

    return img

def bgr2rgb(img):
    return img[...,::-1]

def load_color_img(path, normalize_to_zero_one=True):
    if normalize_to_zero_one:
        return np.array( bgr2rgb(cv.imread(path)), dtype=np.float32 ) / 255.0
    else:
        return np.array(bgr2rgb(cv.imread(path)), dtype=np.float32)

def load_gray_img(path, normalize_to_zero_one=True):
    if normalize_to_zero_one:
        return np.array( cv.imread(path, 0), dtype=np.float32 ) / 255.0
    else:
        return np.array( cv.imread(path, 0), dtype=np.float32 )
