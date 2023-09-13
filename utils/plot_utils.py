
import cv2
import os
from PIL import Image
import numpy as np

def draw_kps2d(image, kps2d):
    '''
    image: cv2 image (or array like cv2 image)
    kps2s: array of shape [n_kps, 2]
    '''

    for x,y in kps2d:
        if not np.isnan(x):
            image = cv2.circle(image, 
                            (round(x), round(y)),
                                1,
                                (0, 0, 255),
                                    2)
    return image


if __name__ == '__main__':
    #test for dev
    pass