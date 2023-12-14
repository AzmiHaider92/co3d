import shutil
from typing import List
import cv2
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from co3d.dataset.rotation import compute_rotation_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pytorch3d
from pytorch3d.implicitron.dataset.dataset_base import FrameData
import torch
from pytorch3d.utils import opencv_from_cameras_projection

from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from visualize.visualize_cameras import read_transforms, visualize_transforms


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])



def process_image(image_path, img_target):
    img = cv2.imread(image_path)
    cv2.imwrite(img_target, img)
    return True


def sort_frames(out):
    frames = out['frames']
    #new_frames = frames.copy()
    num_frames = len(frames)
    frames_paths = [frame['file_path'] for frame in frames]
    sorted_indices = sorted(range(len(frames_paths)), key=lambda k: frames_paths[k])
    new_frames = [frames[sorted_indices[i]] for i in range(num_frames)]
    out['frames'] = new_frames


def train_test_split(out, source_dir, train_target_dir, test_target_dir):
    train_out = out.copy()
    test_out = out.copy()
    folder_masks = source_dir.replace("images", "masks")
    train_frames = []
    test_frames = []

    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(test_target_dir, exist_ok=True)

    for i, frame in enumerate(out['frames']):
        img = os.path.join(source_dir, frame['file_path'].split('/')[-1])
        img_mask = os.path.join(folder_masks, frame['file_path'].split('/')[-1].replace("jpg","png"))
        if i % 2 == 0: # to train
            frame['file_path'] = './train/r_' + str(int(i/2))
            shutil.copyfile(img, os.path.join(train_target_dir, 'r_' + str(int(i / 2)) + '.png'))
            shutil.copyfile(img_mask, os.path.join(train_target_dir, 'rmask_' + str(int(i / 2)) + '.png'))

            train_frames.append(frame)
        else: # to test
            frame['file_path'] = './test/r_' + str(int(i/2))
            shutil.copyfile(img, os.path.join(test_target_dir, 'r_' + str(int(i / 2)) + '.png'))
            shutil.copyfile(img_mask, os.path.join(test_target_dir, 'rmask_' + str(int(i / 2)) + '.png'))

            test_frames.append(frame)
        #process_image(s, t)
    train_out['frames'] = train_frames
    test_out['frames'] = test_frames
    return train_out, test_out


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom


def co3d_annotation_to_opencv_pose(frame_data: FrameData):
    # this extracts the instrinsics and extrinsics from co3d

    h, w = frame_data.image.size

    camera = pytorch3d.renderer.cameras.PerspectiveCameras(focal_length=torch.tensor(frame_data.viewpoint.focal_length).unsqueeze(0),
                                                           principal_point=torch.tensor(frame_data.viewpoint.principal_point).unsqueeze(0),
                                                           in_ndc=True,
                                                           R=torch.tensor(frame_data.viewpoint.R).unsqueeze(0),
                                                           T=torch.tensor(frame_data.viewpoint.T).unsqueeze(0),
                                                           image_size=torch.tensor((h, w)).unsqueeze(0)
                                                           )


    r, t, k = opencv_from_cameras_projection(camera, torch.tensor([[h, w]]))
    # Extract c2w (OpenCV-style camera-to-world transformation, extrinsics).
    R = r[0]
    t = t[0]
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    c2w = np.linalg.inv(m)
    c2w[0:3, 0] *= -1
    #c2w = c2w[:, [1, 0, 2, 3]]  # swap y and z
    c2w[2, :] *= -1  # flip whole world upside down

    R = np.asarray(frame_data.viewpoint.R).T  # note the transpose here
    T = np.asarray(frame_data.viewpoint.T)
    pose = np.concatenate([R, T[:, None]], 1)
    pose = np.diag([-1, -1, 1]).astype(np.float32) @ pose  # flip the direction of x,y axis
    c2w = np.eye(4)
    c2w[:3, :] = pose
    c2w = np.linalg.inv(c2w)
    '''
    m[:3,:3] = -R
    m[:3,3] = t
    c2w = np.linalg.inv(m)
    c2w[0:3, 2] *= -1  # flip the y and z axis
    c2w[0:3, 1] *= -1
    c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
    c2w[2, :] *= -1  # flip whole world upside down
    '''
    # Extract K (camera intrinsics).
    k = np.array(k[0], dtype=np.float)
    #k[:2] /= torch.tensor([w, h])[:, None]

    angle_x = math.atan(w / (k[0, 0] * 2)) * 2
    angle_y = math.atan(h / (k[1, 1] * 2)) * 2

    # return
    return k, c2w, angle_x, angle_y


def calculate_transform(entries):
    # all entries belong to same scene
    up = np.zeros(3)
    out = {}
    h, w = float(entries[0].image.size[0]), float(entries[0].image.size[1])
    out["h"] = h
    out["w"] = w
    out["frames"] = []
    angle_x, angle_y, fl_x, fl_y, cx, cy = [], [], [], [], [], []
    centroid = np.zeros(3)

    for i in range(len(entries)):
        entry = entries[i]
        intrinsic, extrinsic, a_x, a_y = co3d_annotation_to_opencv_pose(entry)

        angle_x.append(a_x)
        angle_y.append(a_y)
        fl_x.append(intrinsic[0, 0])
        fl_y.append(intrinsic[1, 1])
        cx.append(intrinsic[0, 2])
        cy.append(intrinsic[1, 2])

        centroid += np.array(extrinsic[0:3, 3].data)

        frame = {"file_path": entry.image.path, "transform_matrix": extrinsic}
        out["frames"].append(frame)
        up += np.array(extrinsic[0:3, 1].data)

    out["camera_angle_x"] = np.mean(angle_x)
    out["camera_angle_y"] = np.mean(angle_y)
    out["fl_x"] = np.mean(fl_x)
    out["fl_y"] = np.mean(fl_y)
    out["cx"] = np.mean(cx)
    out["cy"] = np.mean(cy)
    out["aabb_scale"] = 16
    nframes = len(entries)

    '''
    centroid *= 1 / nframes
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis
    out["A"] = np.matmul(R, out["A"])  # rotate up to be the z axis
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    # print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= np.pad(totp[:2], (0,1), 'constant')
    out["A"][0:3, 3] -= np.pad(totp[:2], (0, 1), 'constant')
    
    '''
    # scale to nerf size
    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4. / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    return out


if __name__ == "__main__":
    category_frame_annotations = load_dataclass_jgzip(
        r"C:\Users\azmih\Desktop\Projects\datasets\CO3d\hydrant\hydrant_000\frame_annotations.jgz", List[FrameAnnotation]
    )
    #category_sequence_annotations = load_dataclass_jgzip(
    #    r"C:\Users\azmih\Desktop\Projects\datasets\CO3d\vase\vase000\sequence_annotations.jgz", List[SequenceAnnotation]
    #)

    scenes_entries = {}
    for entry in category_frame_annotations:
        sequence_name = entry.sequence_name
        if sequence_name not in scenes_entries:
            scenes_entries[sequence_name] = []
        scenes_entries[sequence_name].append(entry)

    scenes_dir = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\hydrant_one_align_sanity"
    scenes = os.listdir(scenes_dir)
    for scene in tqdm(scenes):
        #scene = '106_12677_24990'
        s = scenes_entries[scene]

        print(f"======={scene}=======:  #images {len(s)}")

        target_dir = os.path.join(scenes_dir, scene)
        images_dir = os.path.join(scenes_dir, scene, 'images')
        if not os.path.exists(images_dir):
            continue

        train_transform_json_path = os.path.join(target_dir, 'transforms_train.json')
        test_transform_json_path = os.path.join(target_dir, 'transforms_test.json')

        if os.path.exists(train_transform_json_path):
            os.remove(train_transform_json_path)
        if os.path.exists(test_transform_json_path):
            os.remove(test_transform_json_path)
        if os.path.exists(os.path.join(target_dir, 'train')):
            shutil.rmtree(os.path.join(target_dir, 'train'))
        if os.path.exists(os.path.join(target_dir, 'test')):
            shutil.rmtree(os.path.join(target_dir, 'test'))

        # alignment matrix
        pcd_file = os.path.join(scenes_dir, scene, "pointcloud.ply")
        Alignment_matrix = compute_rotation_matrix(pcd_file)
        Alignment_matrix_expanded = np.eye(4)
        Alignment_matrix_expanded[:-1, :-1] = Alignment_matrix

        out = calculate_transform(s)
        out["alignment_matrix"] = Alignment_matrix_expanded.tolist()

        #out = calculate_transform_new(s)
        sort_frames(out)
        train_out, test_out = train_test_split(out, images_dir, os.path.join(target_dir, 'train'),
                                           os.path.join(target_dir, 'test'))

        with open(train_transform_json_path, "w") as outfile:
            json.dump(train_out, outfile, indent=2)
        with open(test_transform_json_path, "w") as outfile:
            json.dump(test_out, outfile, indent=2)


        if os.path.exists(os.path.join(target_dir, 'images')):
            shutil.rmtree(os.path.join(target_dir, 'images'))
        if os.path.exists(os.path.join(target_dir, 'masks')):
            shutil.rmtree(os.path.join(target_dir, 'masks'))
        if os.path.exists(os.path.join(target_dir, 'depths')):
            shutil.rmtree(os.path.join(target_dir, 'depths'))
        if os.path.exists(os.path.join(target_dir, 'depth_masks')):
            shutil.rmtree(os.path.join(target_dir, 'depth_masks'))
        #if os.path.exists(os.path.join(target_dir, 'pointcloud.ply')):
        #    os.remove(os.path.join(target_dir, 'pointcloud.ply'))


    '''
    transforms_path = os.path.join(scenes_dir, scene, "transforms_train.json")
    transforms_annot_path = train_transform_json_path
    Transforms, cameraPositions = read_transforms(transforms_path)
    Transforms2, cameraPositions2 = read_transforms(transforms_annot_path)
    v1 = visualize_transforms(Transforms, cameraPositions)
    v2 = visualize_transforms(Transforms2, cameraPositions2)
    plt.show()

    print("please work")

    
    for i, scene in enumerate(scenes_entries):
        if scene !='62_4316_10771':
            continue
        print(i)
        s = scenes_entries[scene]
        print(f"======={scene}=======:  #images {len(s)}")

        target_dir = os.path.join(scenes_dir, scene)
        images_dir = os.path.join(target_dir, 'images')


        train_transform_json_path = os.path.join(target_dir, 'transforms_train.json')
        test_transform_json_path = os.path.join(target_dir, 'transforms_test.json')

        #if os.path.exists(train_transform_json_path) and os.path.exists(test_transform_json_path):
        #    continue

        out = calculate_transform(s)
        sort_frames(out)
        train_out, test_out = train_test_split(out, images_dir, os.path.join(target_dir, 'train'),
                                               os.path.join(target_dir, 'test'))

        with open(train_transform_json_path, "w") as outfile:
            json.dump(train_out, outfile, indent=2)
        with open(test_transform_json_path, "w") as outfile:
            json.dump(test_out, outfile, indent=2)

        print("please work")
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    out = calculate_transform(entries)
    for f in out['frames']:
        transform = np.array(f['transform_matrix'])
        x, y, z = transform[0,3], transform[1,3], transform[2,3]
        ax.scatter(x, y, z,  color='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    end=1
    '''







