import json
import math
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
from tqdm import tqdm


def rotationalMatrix(alpha, beta):
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(beta), -math.sin(beta)],
                   [0, math.sin(beta), math.cos(beta)]])
    Ry = np.array([[math.cos(beta), 0, math.sin(beta)],
                   [0, 1, 0],
                   [-math.sin(beta), 0, math.cos(beta)]])
    Rz = np.array([[math.cos(alpha), -math.sin(alpha), 0],
                   [math.sin(alpha), math.cos(alpha), 0],
                   [0, 0, 1]])
    return Rx, Ry, Rz


def compute_rotation_matrix(ply_file_path):
    pcd = o3d.io.read_point_cloud(ply_file_path)  # Read the point cloud
    point_cloud_in_numpy = np.asarray(pcd.points)

    # BBOX
    x1, x2 = np.min(point_cloud_in_numpy[:, 0]), np.max(point_cloud_in_numpy[:, 0])
    y1, y2 = np.min(point_cloud_in_numpy[:, 1]), np.max(point_cloud_in_numpy[:, 1])
    z1, z2 = np.min(point_cloud_in_numpy[:, 2]), np.max(point_cloud_in_numpy[:, 2])

    print(f" BBOX = [[{x1},{y1},{z1}]  -  [{x2},{y2},{z2}]]")

    N = 10000
    indx = np.random.randint(point_cloud_in_numpy.shape[0], size=N)
    ps = point_cloud_in_numpy[indx, :]

    # PCA
    ps_centered = ps - ps.mean(axis=0)
    U, S, Vt = np.linalg.svd(ps_centered)
    #pca_points = ps_centered @ Vt.T

    max_pca = Vt.T[:, 0]
    alpha = math.atan2(max_pca[1], max_pca[0])
    beta = math.atan2(math.sqrt(max_pca[1] ** 2 + max_pca[0] ** 2), max_pca[2])
    _, Ry, Rz = rotationalMatrix(-alpha, -beta)

    #pc2 = Ry @ Rz @ point_cloud_in_numpy.T
    #pc2 = pc2.T

    return Ry @ Rz


def add_to_json(json_path, R, near, far):
    f = open(json_path, 'r')
    meta = json.load(f)
    f.close()
    updict = {"near_depth": near, "far_depth": far, "alignment_matrix": R.tolist()}
    updict.update(meta)
    with open(json_path, "w") as outfile:
        json.dump(updict, outfile, indent=2)


def near_far_depth(scene):
    depth_dir = os.path.join(scene, "depths")
    depth_masks_dir = os.path.join(scene, "depth_masks")
    all_max, all_min = 0, np.inf
    for img in os.listdir(depth_dir):
        with Image.open(os.path.join(depth_dir, img)) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        with Image.open(os.path.join(depth_masks_dir, img.replace(".jpg.geometric", ""))) as pil_im:
            mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
        # dm = cv2.imread(os.path.join(depth_masks, img.replace(".jpg.geometric", "")), cv2.IMREAD_UNCHANGED)

        # apply mask
        masked = depth * mask
        # plt.figure()
        # plt.imshow(masked)
        # plt.show()
        masked = masked[masked != 0]
        try:
            far, near = np.max(masked) / 10, np.min(masked) / 10
        except:
            far, near = 0, np.inf
        if far > all_max:
            all_max = far
        if near < all_min:
            all_min = near
    return all_min, all_max


def check_scene(json_path):
    f = open(json_path, 'r')
    meta = json.load(f)
    f.close()
    if "alignment_matrix" in meta.keys():
        return True


if __name__ == "__main__":

    main_dir = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\hydrant_one_align"
    scenes = os.listdir(main_dir)

    print(f"Number of scenes in original dataset: {len(scenes)} -----------")
    failed_scenes = []
    for i, scene in enumerate(tqdm(scenes)):
        #
        ply_file = os.path.join(main_dir, scene, "pointcloud.ply")
        if check_scene(os.path.join(main_dir, scene, f"transforms_train.json")) and \
            check_scene(os.path.join(main_dir, scene, f"transforms_test.json")):
            continue
        try:
            R = compute_rotation_matrix(ply_file)
        except:
            print("fail to compute R")
            failed_scenes.append(scene)
            continue
        near, far = near_far_depth(os.path.join(main_dir, scene))
        add_to_json(os.path.join(main_dir, scene, f"transforms_train.json"), R, near, far)
        add_to_json(os.path.join(main_dir, scene, f"transforms_test.json"), R, near, far)

    print(failed_scenes)
    stop=1