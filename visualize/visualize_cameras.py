import os
import numpy as np
import json
import matplotlib.pyplot as plt

coord_trans = np.diag([1, -1, -1, 1])


def read_transforms(transforms_path):
    poses = []
    paths = []
    A = np.eye(4)
    if os.path.isdir(transforms_path):
        pose_files = os.listdir(transforms_path)
        for pose_file in pose_files:
            c2w = np.loadtxt(os.path.join(transforms_path, pose_file),
                                      dtype=np.float32).reshape(4, 4)
            #c2w = c2w @ coord_trans
            poses.append(c2w)
    else: # json file
        with open(transforms_path, 'r') as f:
            data = json.load(f)
            A = np.array(data["alignment_matrix"])
            frames = data['frames']
            for frame in frames:
                c2w = np.array(frame['transform_matrix'])
                #c2w = np.linalg.inv(c2w)
                poses.append(c2w)
                paths.append(frame['file_path'])

    return poses, A, paths

from visualize.camera_visualizer import CameraPoseVisualizer
def visualize_transforms(Transforms, cameraPositions):
    cameraPositions = np.array(cameraPositions)
    m=4
    visualizer = CameraPoseVisualizer([-m, m],
                                      [-m, m],
                                      [-m, m])
    max_frame_length = len(Transforms)
    for idx_frame, T in enumerate(Transforms):
        visualizer.extrinsic2pyramid(T, plt.cm.rainbow(idx_frame / max_frame_length), 10)

    visualizer.colorbar(max_frame_length)
    return visualizer


if __name__ == "__main__":
    transforms_path = r"C:\Users\azmih\Desktop\Projects\datasets\CO3d\vase\vase_all\62_4316_10771\transforms_train.json"
    transforms_annot_path = r"C:\Users\azmih\Desktop\Projects\datasets\CO3d\vase\vase_all\62_4316_10771\transforms_train2.json"
    Transforms, cameraPositions, _ = read_transforms(transforms_path)
    Transforms2, cameraPositions2, _ = read_transforms(transforms_annot_path)
    visualize_transforms(Transforms, cameraPositions)
    visualize_transforms(Transforms2, cameraPositions2)
    plt.show()

    end=1