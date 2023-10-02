import os
import cv2
import glob
import numpy as np
import utils.plot_utils
from tqdm import tqdm

if __name__ == "__main__":
    #test for dev
    wcam_sd_dir = "/media/basic_data/saud_sd"
    kps_2d_files = [f for f in sorted(os.listdir("/media/basic_data/saud_sd")) \
                    if "_kps_2d.npy" in f]

    for f in tqdm(kps_2d_files):
        data_point = f.split("_")[0]
        kps_2d = np.load(os.path.join(wcam_sd_dir, f))
        source = np.zeros( (288, 288, 3))
        source = utils.plot_utils.draw_kps2d_img(source, kps_2d, add_text=False)
        source = cv2.resize(source, (256, 256))    
        cv2.imwrite(os.path.join(wcam_sd_dir, data_point + "_control.png"), source)


    # file_list  = os.listdir(wcam_sd_dir)

    # kps_vis = [f for f in file_list if "kps_vis" in f]
    

    # index = 0
    # data_point =   kps_vis[index].split("_")[0]
    # kps_vis_0 = np.load(os.path.join(wcam_sd_dir, kps_vis[index]))
    # kps_2d = np.load(os.path.join(wcam_sd_dir, f'{data_point}_kps_2d.npy'))
    # kps_3d = np.load(os.path.join(wcam_sd_dir, f'{data_point}_kps_3d.npy'))

    # im = cv2.imread(os.path.join(wcam_sd_dir, f'{data_point}_0_generated.png'))

    # print(im.shape, kps_vis_0.shape, kps_2d.shape, kps_3d.shape)
    # im = utils.plot_utils.draw_kps2d(im, kps_2d, show_indices=True)
    # cv2.imwrite("samples/test.png", im)
    # # print(data_point,kps_2d, kps_vis_0)