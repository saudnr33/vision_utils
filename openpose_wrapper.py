from annotators.openpose import OpenposeDetector
import argparse
import cv2
from utils.image_utils import HWC3, resize_image


def get_args():
    parser = argparse.ArgumentParser(description='Image + openpose')
    parser.add_argument('--input_image', type=str,default=None,
                        help='Path to input image')
    parser.add_argument('--detect_resolution', type=int,default=512,
                    help=' ')
    args = parser.parse_args()
    return args


def proccess_single_image(input_image, detect_resolution):
    '''
    This is mainly used to show user how to load images.
    '''
    preprocessor = OpenposeDetector()

    #** Load with open cv **#
    input_image = cv2.imread(input_image) # H, W, C (BGR)
    input_image = HWC3(input_image)
    input_image =  input_image[:, :, ::-1] # H, W, C (RGB)
    

    detected_map = preprocessor(resize_image(input_image, detect_resolution), hand_and_face=True) 
    detected_map = HWC3(detected_map)

    return input_image[:, :, ::-1] , detected_map[:, :, ::-1] 


if __name__ == "__main__":


    args = get_args()
    input_image, detected_map = proccess_single_image(**vars(args))

    cv2.imwrite("samples/im1.png" , input_image)
    cv2.imwrite("samples/im2.png" , detected_map)

