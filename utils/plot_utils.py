
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

def draw_finger_line(image, kps2d, colors):
    core = kps2d[-1]
    for i in range(5):
        for j in range(3):
            x1, y1 = kps2d[4*i + j]
            x2, y2 = kps2d[4*i + j + 1]
            if not (np.isnan(x1) or np.isnan(x2)):
                start_point = (round(x1), round(y1) )
                end_point = (round(x2), round(y2))
                # Green color in BGR
                color = colors[4*i + j]
                # Line thickness of 9 px
                thickness = 1

                image = cv2.line(image,
                                  start_point,
                                    end_point,
                                      color,
                                        thickness)
        
        if not (np.isnan(core[0]) or np.isnan(x2)):
                start_point = (round(core[0]), round(core[1]))
                end_point = (round(x2), round(y2))
                color = colors[4*i + 3]
                thickness = 2
                image = cv2.line(image,
                                  start_point,
                                    end_point,
                                      color,
                                        thickness)
    return image


def draw_kps2d_img(image, kps2d, add_text = False):
    '''
    image: cv2 image (or array like cv2 image)
    kps2s: array of shape [n_kps, 2]
    '''
    colors = [[250, 110, 110],
                [242, 103, 113],
                [233, 97, 117],
                [224, 92, 119],
                [214, 87, 121],
                [204, 83, 123],
                [193, 79, 124],
                [182, 76, 125],
                [171, 73, 125],
                [159, 71, 125],
                [147, 68, 124],
                [135, 66, 122],
                [123, 64, 120],
                [111, 61, 117],
                [99, 59, 114],
                [87, 56, 109],
                [76, 54, 105],
                [64, 51, 100],
                [53, 48, 94],
                [42, 45, 88]]
    core = (255, 126, 0)
    for i, (x,y) in enumerate(kps2d):
        if not np.isnan(x):
            if i ==20:
                color = core
            else:
                color = colors[i] 
            image = cv2.circle(image, 
                            (round(x), round(y)),
                                3,
                                color,
                                    4)
            
            if add_text:
                org = (round(x), round(y))
                fontScale = 0.5
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                image = cv2.putText(image,
                                     f'{i}',
                                       org, 
                                       font, 
                                       fontScale,
                                         color,
                                           thickness,
                                             cv2.LINE_AA)  
    image = draw_finger_line(image, kps2d, colors)           
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