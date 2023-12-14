import os.path

import numpy as np
import matplotlib.pyplot as plt
from rotation import compute_rotation_matrix
from visualize.visualize_cameras import read_transforms, visualize_transforms
import open3d as o3d
import cv2
blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

if __name__ == "__main__":
    input_file = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\hydrant_four_align\106_12677_24990\pointcloud.ply"
    #Alignment_matrix = compute_rotation_matrix(input_file)
    #Alignment_matrix_expanded = np.eye(4)
    #Alignment_matrix_expanded[:-1, :-1] = Alignment_matrix
    #Alignment_matrix_expanded = np.linalg.inv(Alignment_matrix_expanded)


    transforms_path = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\hydrant_four_align\106_12677_24990\transforms_train.json"
    Transforms, A, img_paths = read_transforms(transforms_path)
    #Transforms = [T @ blender2opencv for T in Transforms]
    Aligned_transforms = [A @ T for T in Transforms]

    # prepare points (the hydrant points)
    pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud
    points_colors = np.asarray(pcd.colors)
    point_cloud_in_numpy = np.asarray(pcd.points)
    N = 10000
    indx = np.random.randint(point_cloud_in_numpy.shape[0], size=N)
    raw_points = point_cloud_in_numpy[indx, :]
    points_colors = points_colors[indx, :]
    #raw_points = (A[:-1, :-1] @ raw_points.T).T

    # original
    img_idx = 74
    visualizer = visualize_transforms([Transforms[img_idx]], [])
    visualizer.ax.scatter(raw_points[:, 0], raw_points[:, 1], raw_points[:, 2], c=points_colors, s=1.2)
    visualizer.title("original")

    plt.figure()
    img = cv2.cvtColor(cv2.imread(
        os.path.join(r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\hydrant_four_align\106_12677_24990",img_paths[img_idx]+'.png'))
        , cv2.COLOR_BGR2RGB)
    #img = np.fliplr(img)
    plt.imshow(img)


    # aligned
    Aligned_points = (A[:-1, :-1] @ raw_points.T).T
    visualizer = visualize_transforms([Aligned_transforms[img_idx]], [])
    visualizer.ax.scatter(Aligned_points[:,0], Aligned_points[:,1], Aligned_points[:,2],c=points_colors, s=2)
    visualizer.title("aligned")



    print("show me the money")




