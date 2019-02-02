# modified from https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
# Ported from: http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
import math
import sys

import cv2
import numpy as np


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def simplest_cb(image, percent, verbose=False):
    assert image.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(image)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high percentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        if verbose:
            print("Lowval: " + str(low_val))
            print("Highval: " + str(high_val))

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    #img = cv2.imread('../Data/PP_test/CSt_No-Porous/sq_77_10X PP 3.jpg')
    #img = cv2.imread('../Data/PP_test/CSt_No-Porous/sq_44_10X PP 2.jpg')
    out = simplest_cb(img, 1)
    cv2.imshow("before", img)
    cv2.imshow("after", out)
    cv2.waitKey(0)
