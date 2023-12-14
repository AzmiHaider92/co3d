import math
import random
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


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


# Read .ply file
input_file = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\hydrant_multiple\116_13651_28370_pointcloud.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud


# Visualize the point cloud within open3d
vis = o3d.visualization
vis.Visualizer.set_full_screen = True
show_coordinate_frame = True
vis.draw_geometries([pcd])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format.
point_cloud_in_numpy = np.asarray(pcd.points)

# BBOX
x1,x2 = np.min(point_cloud_in_numpy[:, 0]) , np.max(point_cloud_in_numpy[:, 0])
y1,y2 = np.min(point_cloud_in_numpy[:, 1]) , np.max(point_cloud_in_numpy[:, 1])
z1,z2 = np.min(point_cloud_in_numpy[:, 2]) , np.max(point_cloud_in_numpy[:, 2])

print(f" BBOX = [[{x1},{y1},{z1}]  -  [{x2},{y2},{z2}]]")

# maybe choose random N points
N = 10000
indx = np.random.randint(point_cloud_in_numpy.shape[0], size=N)
ps = point_cloud_in_numpy[indx, :]

# PCA
ps_centered = ps - ps.mean(axis=0)
U, S, Vt = np.linalg.svd(ps_centered)
pca_points = ps_centered @ Vt.T

max_pca = Vt.T[:, 0]
alpha = math.atan2(max_pca[1], max_pca[0])
beta = math.atan2(math.sqrt(max_pca[1]**2 + max_pca[0]**2), max_pca[2])
_, Ry, Rz = rotationalMatrix(-alpha, -beta)
_, Ry2, Rz2 = rotationalMatrix(alpha, beta)
T1 = Ry @ Rz
T2 = Ry2 @ Rz2

print(T1)

pc2 = Ry @ Rz @ point_cloud_in_numpy.T
pc2 = pc2.T


# second time
#U, S, Vt = np.linalg.svd(pc2)
#print(Vt.T[:, 0])

#angle = math.acos(np.dot(np.array([0,0,1]), Vt.T[:,0]))
#Rz = np.array([[math.cos(angle), math.sin(angle), 0],
#               [-math.sin(angle), math.cos(angle), 0],
#               [0, 0, 1]])



# geometry is the point cloud used in your animaiton
'''
pcd_before = o3d.geometry.PointCloud()
pcd_before.points = o3d.utility.Vector3dVector(ps)
o3d.visualization.draw_geometries([pcd_before])



'''
pcd_after = o3d.geometry.PointCloud()
pcd_after.points = o3d.utility.Vector3dVector(pc2)
pcd_after.colors = pcd.colors

# visualize
vis = o3d.visualization.Visualizer()
vis.create_window()
o3d.visualization.draw_geometries([pcd_after])

output_file = input_file.replace("point","alignedpoint")
o3d.io.write_point_cloud(output_file, pcd_after) # Write the point cloud


stop=1