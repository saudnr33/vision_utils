import os
import cv2
import glob
import numpy as np
import utils.plot_utils


if __name__ == "__main__":
    #test for dev
    wcam_sd_dir = "/media/basic_data/ht_datasets_ml_datasets_wcam_sd_v1.5/wcam_sd"
    file_list  = os.listdir(wcam_sd_dir)

    kps_vis = [f for f in file_list if "kps_vis" in f]
    

    index = 0
    data_point =   kps_vis[index].split("_")[0]
    kps_vis_0 = np.load(os.path.join(wcam_sd_dir, kps_vis[index]))
    kps_2d = np.load(os.path.join(wcam_sd_dir, f'{data_point}_kps_2d.npy'))
    im = cv2.imread(os.path.join(wcam_sd_dir, f'{data_point}_0_generated.png'))

    print(im.shape)
    im = utils.plot_utils.draw_kps2d(im, kps_2d, show_indices=True)
    cv2.imwrite("samples/test.png", im)
    # print(data_point,kps_2d, kps_vis_0)