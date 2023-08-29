import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm






def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        required=True,
        nargs='*',
        type=str,
        default=None,
        help="list of your images folder (e.g. path/to/dir1 path/to/dir2 ...)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="frames per second, default = 25.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        default=25,
        help="output directory including video name (e.g. path/to/video.mp4)",
    )
    

    args = parser.parse_args()
    return args





def multiple_veiwer(datasets, fps = 25, output = None):
    if output == None:
        print('Please specify the output location, otherwise the video will saved at the current working directory video.avi!')
        output = "video.avi"
    
    num_datasets = len(datasets)
    
    paths = []
    for dataset in datasets:
        img_names = sorted(os.listdir(dataset))
        paths.append(img_names)


    # Print data identifiers:
    h, w = cv2.imread(os.path.join(datasets[0], paths[0][0]), cv2.IMREAD_ANYDEPTH).shape
    video_size = (h, w * num_datasets)

    

    video_writer = cv2.VideoWriter(output,
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    fps,
                                    (w * num_datasets, h))

    theater = np.zeros(video_size).astype(np.uint8)

    for i in tqdm(range(min([len(dataset) for dataset in paths]))):
        for j in range(len(datasets)):

            img = cv2.imread(os.path.join(datasets[j], paths[j][i]), cv2.IMREAD_ANYDEPTH)
            theater[:, j*w: (j + 1)*w] = img



        video_writer.write(cv2.cvtColor(theater, cv2.COLOR_GRAY2BGR))
    video_writer.release()



if __name__ == "__main__":
    args = get_args()
    multiple_veiwer(**vars(args))
    # main(args)
