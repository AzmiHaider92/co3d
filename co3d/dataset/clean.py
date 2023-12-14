import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    main_dir = r"C:\Users\azmih\Desktop\Projects\datasets\CO3d\vase\vase_all"
    corrupt_dir = r"C:\Users\azmih\Desktop\Projects\datasets\CO3d\vase\corrupt_vases"
    scenes = os.listdir(main_dir)
    thresh = 50

    print(f"Number of scenes in original dataset: {len(scenes)} -----------")
    have_poses = np.zeros((len(scenes)))
    num_imgs = np.zeros((len(scenes)))
    for i, scene in enumerate(scenes):
        # has poses
        if os.path.exists(os.path.join(main_dir, scene, "transforms_train.json")) and \
                os.path.exists(os.path.join(main_dir, scene, "transforms_test.json")):
            have_poses[i] = 1

        # num images
        num_imgs[i] = len(os.listdir(os.path.join(main_dir, scene, "train")))

        if have_poses[i] == 0 or num_imgs[i] < thresh:
            shutil.move(os.path.join(main_dir, scene), os.path.join(corrupt_dir, scene))


    # statistics
    print(f"Number of scenes with no poses: {len(scenes) - np.sum(have_poses)} ####")
    plt.figure()
    plt.hist(num_imgs)
    plt.show()

    num_imgs_larger_than_thresh = np.sum(np.array(num_imgs>=thresh, dtype=int))
    num_scenes_less_than_thresh_and_nopose = np.sum(np.array(num_imgs<thresh, dtype=int) * (1-have_poses))
    end=1