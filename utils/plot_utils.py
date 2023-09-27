
import cv2
import os
from PIL import Image
import numpy as np

def draw_kps2d(image, kps2d, show_indices = False):
    '''
    image: cv2 image (or array like cv2 image)
    kps2s: array of shape [n_kps, 2]
    '''

    for i, (x,y) in enumerate(kps2d):
        if not np.isnan(x):
            image = cv2.circle(image, 
                            (round(x), round(y)),
                                1,
                                (0, 0, 255),
                                    2)
            if show_indices:
                image = cv2.putText(image, str(i), (round(x), round(y)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                     0.5,
                                       (255, 0, 0),
                                         1,
                                           cv2.LINE_AA)
    return image


def image_overlayer(im_source, im_overlay, remove_black_background = False):
    '''
    im_source: cv2 image
    im_overlay: cv2 image
    '''
    if remove_black_background:
        im = im_source.copy()
        background = np.sum(im_overlay, axis = 2) != 0
        im[:, :, 0][background] = 0
        im[:, :, 1][background] = 0
        im[:, :, 2][background] = 254
    else:
        im = cv2.addWeighted(im_source,0.5,im_overlay,0.5,0)
    return im

if __name__ == '__main__':
    #test for dev
    
    im_source = cv2.imread('/home/MAGICLEAP/salrasheed/Pictures/trop.png')
    im_overlay = cv2.imread('/home/MAGICLEAP/salrasheed/Pictures/trop_control.png')
    im = image_overlayer(im_source, im_overlay, remove_black_background= True)
    cv2.imwrite("samples/overlay.png", im)
    pass