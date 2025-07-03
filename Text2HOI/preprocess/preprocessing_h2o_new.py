import os
import os.path as osp

import glob
import numpy as np
import tqdm
import time
import json
import trimesh
import pickle
from collections import Counter

import torch
from PIL import Image
import cv2
from moviepy.editor import ImageSequenceClip

from constants.h2o_constants import (
    h2o_obj_name, 
    action_list, 
    present_participle, 
    third_verb, 
    passive_verb, 
)
from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.utils.file import load_config
from lib.utils.frame import align_frame
from lib.utils.proc_h2o import (
    get_data_path_h2o, 
    process_hand_pose_h2o,
    process_hand_trans_h2o,
    process_text, 
)
from lib.utils.proc import (
    get_contact_info, 
    transform_hand_to_xdata,
    transform_xdata_to_joints, 
    transform_obj_to_xdata, 
    farthest_point_sample, 
)

import h5py

def visuliaze_joints(j_lhand, j_rhand, image_list, intrinsic_matrix, resolution, action_name):
    """
    Visualize hand joints on images and create a video
    Args:
        j_lhand, j_rhand: hand joints in camera coordinates (T, 25, 3)
        image_list: list of images
        intrinsic_matrix: 3x3 camera intrinsic matrix
        resolution: original resolution (width, height)
    """

    
    # Get current image size
    current_h, current_w = image_list[0].shape[:2]
    
    # Scale intrinsic matrix according to image resize
    scale_w = current_w / resolution[0]
    scale_h = current_h / resolution[1]
    scaled_intrinsic = intrinsic_matrix.copy()
    scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
    scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy
    
    frames = []
    # Hand joint connections for visualization
    connections = [(0,1), (1,2), (2,3), (3,4),  # thumb
                  (0,5), (5,6), (6,7), (7,8),    # index
                  (0,9), (9,10), (10,11), (11,12),  # middle
                  (0,13), (13,14), (14,15), (15,16),  # ring
                  (0,17), (17,18), (18,19), (19,20)]  # pinky
    
    for j_l, j_r, img in zip(j_lhand, j_rhand, image_list):
        # Convert image to BGR if it's RGB
        if len(img.shape) == 3:
            img_draw = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        # Project 3D points to 2D
        # Homogeneous transformation
        j_l_2d = (scaled_intrinsic @ j_l.T).T  # (25, 3)
        j_l_2d = j_l_2d[:, :2] / j_l_2d[:, 2:]  # Perspective division
        j_r_2d = (scaled_intrinsic @ j_r.T).T
        j_r_2d = j_r_2d[:, :2] / j_r_2d[:, 2:]
        
        # Draw joints and connections
        # Left hand (red)
        for joint in j_l_2d:
            cv2.circle(img_draw, (int(joint[0]), int(joint[1])), 5, (0,0,255), -1)
        for conn in connections:
            pt1 = tuple(map(int, j_l_2d[conn[0]]))
            pt2 = tuple(map(int, j_l_2d[conn[1]]))
            cv2.line(img_draw, pt1, pt2, (0,0,255), 1)
            
        # Right hand (blue)
        for joint in j_r_2d:
            cv2.circle(img_draw, (int(joint[0]), int(joint[1])), 5, (255,0,0), -1)
        for conn in connections:
            pt1 = tuple(map(int, j_r_2d[conn[0]]))
            pt2 = tuple(map(int, j_r_2d[conn[1]]))
            cv2.line(img_draw, pt1, pt2, (255,0,0), 1)
        
        cv2.putText(img_draw, action_name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        frames.append(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    
    # Create video using moviepy
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile("hand_joints_visualization.mp4")
    
    return frames
def preprocessing_object():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    objects_folder = glob.glob(osp.join(h2o_config.obj_root, "*"))
    
    obj_pcs = {}
    obj_pc_normals = {}
    point_sets = {}
    obj_path = {}
    for object_folder in tqdm.tqdm(objects_folder):
        object_paths = glob.glob(osp.join(object_folder, "*.obj"))
        for object_path in tqdm.tqdm(object_paths):
            mesh = trimesh.load(object_path, maintain_order=True)
            verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
            normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
            normal = normal / torch.norm(normal, dim=2, keepdim=True)
            point_set = farthest_point_sample(verts, 1024)
            sampled_pc = verts[0, point_set[0]].cpu().numpy()
            sampled_normal = normal[0, point_set[0]].cpu().numpy()
            object_name = object_path.split("/")[-2]
            key = f"{object_name}"
            obj_pcs[key] = sampled_pc
            obj_pc_normals[key] = sampled_normal
            point_sets[key] = point_set[0].cpu().numpy()
            obj_path[key] = "/".join(object_path.split("/")[-2:])
    
    os.makedirs("data/h2o", exist_ok=True)
    with open("data/h2o/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": h2o_obj_name, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)

def preprocessing_data():
    start_time = time.time()
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_root = h2o_config.root
    data_save_path = h2o_config.data_path
        
    object_model = build_object_model(h2o_config.data_obj_pc_path)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=h2o_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=h2o_config.flat_hand)
    rhand_layer = rhand_layer.cuda()

    x_lhand_total = []
    x_rhand_total = []
    j_lhand_total = []
    j_rhand_total = []
    x_obj_total = []
    x_lhand_org_total = [] # root position of left hand
    x_rhand_org_total = [] # root position of right hand
    lcf_idx_total = []
    lcov_idx_total = []
    lchj_idx_total = []
    ldist_value_total = []
    rcf_idx_total = []
    rcov_idx_total = []
    rchj_idx_total = []
    rdist_value_total = []
    is_lhand_total = []
    is_rhand_total = []
    lhand_beta_total = []
    rhand_beta_total = []
    object_idx_total = []
    action_total = []
    action_name_total = []
    nframes_total = []
    subject_total = []
    background_total = []

    image_total = []

    no_inter_action_name = []

    total_sequence_names = []
    total_timestamps = []
    intrinsic_total = []
    resolution_total = []
    '''
        subject, background, object class, cam, 
    '''
    subject = 3
    data_paths = glob.glob(osp.join(data_root, f"subject{subject}_ego", "*", "*", "cam*"))
    #data_paths = glob.glob(osp.join(data_root, "subject2_ego", "*", "*", "cam*"))
    #data_paths = glob.glob(osp.join(data_root, "subject3_ego", "*", "*", "cam*"))
    #data_paths = glob.glob(osp.join(data_root, "subject4_ego", "*", "*", "cam*"))
    data_paths.sort()

    for data_idx, data_path in enumerate(data_paths):
        hand_pose_manos, obj_pose_rts, cam_poses, action_labels, images, intrinsic_matrix, resolution \
            = get_data_path_h2o(data_path)
        # print(get_data_path_h2o(data_path))
        # print('hand_pose_manos', hand_pose_manos)
        # print('obj_pose_rts', obj_pose_rts)
        # # print('cam_poses', cam_poses)
        # print('action_labels', action_labels)
        # # print('images', images)
        # print('intrinsic_matrix', intrinsic_matrix)
        # print('resolution', resolution)
        prev_action = 0

        lhand_pose_list = []
        lhand_beta_list = []
        lhand_trans_list = []
        x_lhand_org_list = []
        rhand_pose_list = []
        rhand_beta_list = []
        rhand_trans_list = []
        x_rhand_org_list = []
        object_rotmat_list = []

        image_list = []
        timestamp_list = []
        t = 0
        for hand_pose_mano, obj_pose_rt, cam_pose, action_label, image in \
            tqdm.tqdm(zip(hand_pose_manos, obj_pose_rts, cam_poses, action_labels, images), desc=f"{data_idx}/{len(data_paths)}", total=len(obj_pose_rts)):
            # print('r')
            hand_pose_mano_data = np.loadtxt(hand_pose_mano)
            obj_pose_rt_data = np.loadtxt(obj_pose_rt)
            extrinsic_matrix = np.loadtxt(cam_pose).reshape(4, 4)
            # print('s')
            action = int(np.loadtxt(action_label))
            img = np.array(Image.open(image))
            img = np.array(Image.fromarray(img).resize((256, 256), Image.LANCZOS))
            # print('a')
            
            if action != prev_action and prev_action != 0:
                if len(object_rotmat_list) > 20:
                    _, obj_pc, _, _ = object_model(int(obj_idx))
                    lcf_idx, lcov_idx, lchj_idx, ldist_value, \
                    rcf_idx, rcov_idx, rchj_idx, rdist_value, \
                    is_lhand, is_rhand = get_contact_info(
                        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                        object_rotmat_list, lhand_layer, rhand_layer,
                        obj_pc, 
                    )
                    if is_lhand == 0 and is_rhand == 0:
                        action_name = action_list[prev_action]
                        no_inter_action_name.append(action_name)
                        lhand_pose_list = []
                        lhand_beta_list = []
                        lhand_trans_list = []
                        x_lhand_org_list = []
                        rhand_pose_list = []
                        rhand_beta_list = []
                        rhand_trans_list = []
                        x_rhand_org_list = []
                        object_rotmat_list = []
                        image_list = []
                        prev_action = action
                        timestamp_list = []
                        # print('b')
                        continue
                    x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
                    x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
                    j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                    j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                    x_obj = transform_obj_to_xdata(object_rotmat_list)
                    x_lhand_total.append(x_lhand)
                    x_rhand_total.append(x_rhand)
                    j_lhand_total.append(np.array(j_lhand))
                    j_rhand_total.append(np.array(j_rhand))
                    x_obj_total.append(x_obj)
                    x_lhand_org_total.append(x_lhand_org_list)
                    x_rhand_org_total.append(x_rhand_org_list)
                    lcf_idx_total.append(lcf_idx)
                    lcov_idx_total.append(lcov_idx)
                    lchj_idx_total.append(lchj_idx)
                    ldist_value_total.append(ldist_value)
                    rcf_idx_total.append(rcf_idx)
                    rcov_idx_total.append(rcov_idx)
                    rchj_idx_total.append(rchj_idx)
                    rdist_value_total.append(rdist_value)
                    is_lhand_total.append(is_lhand)
                    is_rhand_total.append(is_rhand)
                    lhand_beta_total.append(lhand_beta_list)
                    rhand_beta_total.append(rhand_beta_list)
                    object_idx_total.append(int(obj_idx))
                    action_name = make_caption(action_list[prev_action], is_lhand, is_rhand)
                    
                    action_total.append(prev_action)
                    action_name_total.append(action_name)
                    nframes_total.append(len(object_rotmat_list))
                    subject_total.append(data_path.split("/")[3])
                    background_total.append(data_path.split("/")[4])

                    image_total.append(np.array(image_list))

                    total_sequence_names.append(data_path)
                    total_timestamps.append(np.array(timestamp_list))
                    intrinsic_total.append(intrinsic_matrix)
                    resolution_total.append(resolution)


                lhand_pose_list = []
                lhand_beta_list = []
                lhand_trans_list = []
                x_lhand_org_list = []
                rhand_pose_list = []
                rhand_beta_list = []
                rhand_trans_list = []
                x_rhand_org_list = []
                object_rotmat_list = []
                image_list = []
                timestamp_list = []
            
            if action == 0:
                prev_action = action
                # print('c')
                continue

            lhand_trans = hand_pose_mano_data[1:4]
            lhand_pose = hand_pose_mano_data[4:52]
            lhand_beta = hand_pose_mano_data[52:62]

            # transform hand pose to camera coordinate
            left_rotvec = process_hand_pose_h2o(lhand_pose, lhand_trans, extrinsic_matrix)
            lhand_pose[:3] = left_rotvec

            new_left_trans, lhand_origin = process_hand_trans_h2o(lhand_pose, lhand_beta, lhand_trans, extrinsic_matrix, lhand_layer)
            lhand_trans_list.append(new_left_trans)
            lhand_pose_list.append(lhand_pose)
            lhand_beta_list.append(lhand_beta)
            x_lhand_org_list.append(lhand_origin)

            rhand_trans = hand_pose_mano_data[63:66]
            rhand_pose = hand_pose_mano_data[66:114]
            rhand_beta = hand_pose_mano_data[114:124]

            # transform hand pose to camera coordinate
            right_rotvec = process_hand_pose_h2o(rhand_pose, rhand_trans, extrinsic_matrix)
            rhand_pose[:3] = right_rotvec

            new_right_trans, rhand_origin = process_hand_trans_h2o(rhand_pose, rhand_beta, rhand_trans, extrinsic_matrix, rhand_layer)
            rhand_trans_list.append(new_right_trans)
            rhand_pose_list.append(rhand_pose)
            rhand_beta_list.append(rhand_beta)
            x_rhand_org_list.append(rhand_origin)

            obj_idx = obj_pose_rt_data[0]
            object_ext = obj_pose_rt_data[1:].reshape(4, 4)

            new_object_matrix = np.dot(extrinsic_matrix, object_ext)
            object_rotmat_list.append(new_object_matrix)

            image_list.append(img)
            timestamp_list.append(t)
            t += 1
            prev_action = action

            # print('d')
        if len(object_rotmat_list) > 20:
            _, obj_pc, _, _ = object_model(int(obj_idx))
            lcf_idx, lcov_idx, lchj_idx, ldist_value, \
            rcf_idx, rcov_idx, rchj_idx, rdist_value, \
            is_lhand, is_rhand = get_contact_info(
                lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                object_rotmat_list, lhand_layer, rhand_layer,
                obj_pc
            )
            if is_lhand == 0 and is_rhand == 0:
                action_name = action_list[prev_action]
                no_inter_action_name.append(action_name)
                continue
            x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
            x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
            j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
            j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
            x_obj = transform_obj_to_xdata(object_rotmat_list)
            action_name = make_caption(action_list[prev_action], is_lhand, is_rhand)
            print('hand shape', x_lhand.shape, x_rhand.shape)
            print('joint shape', j_lhand.shape, j_rhand.shape)
            print('obj shape', x_obj.shape)
            print('is_lhand', is_lhand, 'is_rhand', is_rhand)
            print('action name', action_name)
            # visuliaze_joints(j_lhand, j_rhand, image_list, intrinsic_matrix, resolution, action_name)
            # raise
            x_lhand_total.append(x_lhand)
            x_rhand_total.append(x_rhand)
            j_lhand_total.append(np.array(j_lhand))
            j_rhand_total.append(np.array(j_rhand))
            x_obj_total.append(x_obj)
            x_lhand_org_total.append(x_lhand_org_list)
            x_rhand_org_total.append(x_rhand_org_list)
            lcf_idx_total.append(lcf_idx)
            lcov_idx_total.append(lcov_idx)
            lchj_idx_total.append(lchj_idx)
            ldist_value_total.append(ldist_value)
            rcf_idx_total.append(rcf_idx)
            rcov_idx_total.append(rcov_idx)
            rchj_idx_total.append(rchj_idx)
            rdist_value_total.append(rdist_value)
            is_lhand_total.append(is_lhand)
            is_rhand_total.append(is_rhand)
            lhand_beta_total.append(lhand_beta_list)
            rhand_beta_total.append(rhand_beta_list)
            object_idx_total.append(int(obj_idx))
            
            
            action_total.append(prev_action)
            action_name_total.append(action_name)
            nframes_total.append(len(object_rotmat_list))
            subject_total.append(data_path.split("/")[3])
            background_total.append(data_path.split("/")[4])
            
            #print('image_list', np.array(image_list).shape)
            image_total.append(np.array(image_list))
            total_sequence_names.append(data_path)
            total_timestamps.append(np.array(timestamp_list))

            intrinsic_total.append(intrinsic_matrix)
            resolution_total.append(resolution)
        
    fps = 5 
    ori_fps = 30
    
    # image_total = np.array(image_total)
    # j_lhand_total = np.array(j_lhand_total)
    # j_rhand_total = np.array(j_rhand_total)
    # total_timestamps = np.array(total_timestamps)

    for i in range(len(image_total)):
        fps_index = list(range(0, len(total_timestamps[i]), ori_fps//fps))
        print('fps_index', fps_index)
        print('total_timestamps[i]', total_timestamps[i])
        image_total[i] = image_total[i][fps_index]
        j_lhand_total[i] = j_lhand_total[i][fps_index]
        j_rhand_total[i] = j_rhand_total[i][fps_index]
        total_timestamps[i] = total_timestamps[i][fps_index]

    print('total_images', len(image_total))
    print('total_instructions', len(action_name_total))
    print('total_lhand_joints', len(j_lhand_total))
    print('total_rhand_joints', len(j_rhand_total))
    print('total_sequence_names', len(total_sequence_names))
    print('total_timestamps', len(total_timestamps))

    # Create directory if it doesn't exist
    os.makedirs("data/h2o/preprocess_data", exist_ok=True)

    # Save data using h5py
    h5_path = f"data/h2o/preprocess_data/h2o_data_{subject}.h5"

    with h5py.File(h5_path, 'w') as f:
        # Create groups for different data types
        images_group = f.create_group('images')
        instructions_group = f.create_group('instructions')
        lhand_joints_group = f.create_group('lhand_joints')
        rhand_joints_group = f.create_group('rhand_joints')
        sequence_names_group = f.create_group('sequence_names')
        timestamps_group = f.create_group('timestamps')
        intrinsic_group = f.create_group('intrinsics')
        resolution_group = f.create_group('resolutions')
        # Store each sequence
        for i in range(len(image_total)):
            # Store images as uint8 to save space
            images_group.create_dataset(f'seq_{i}', data=image_total[i], dtype='uint8', compression='gzip')
            
            # Store instructions as strings
            instructions_group.create_dataset(f'seq_{i}', data=np.array([action_name_total[i]], dtype='S'))
            
            # Store joint data
            lhand_joints_group.create_dataset(f'seq_{i}', data=j_lhand_total[i], compression='gzip')
            rhand_joints_group.create_dataset(f'seq_{i}', data=j_rhand_total[i], compression='gzip')
            
            # Store sequence names
            sequence_names_group.create_dataset(f'seq_{i}', data=np.array([total_sequence_names[i]], dtype='S'))
            
            # Store timestamps
            timestamps_group.create_dataset(f'seq_{i}', data=total_timestamps[i], compression='gzip')

            # Store intrinsic
            intrinsic_group.create_dataset(f'seq_{i}', data=intrinsic_total[i], compression='gzip')

            # Store resolution
            resolution_group.create_dataset(f'seq_{i}', data=resolution_total[i], compression='gzip')
        
        # Store metadata
 
        f.attrs['num_sequences'] = len(image_total)
        f.attrs['fps'] = fps
        f.attrs['original_fps'] = ori_fps
    
    print(f"Data successfully saved to {h5_path}")
    print("Finish:", time.time()-start_time)

def make_caption(action_name, is_lhand, is_rhand):
    if is_lhand == 0 and is_rhand == 0:
        return action_name
    if is_lhand == 1 and is_rhand == 0:
        return f"{action_name} with left hand.".capitalize()
    if is_lhand == 0 and is_rhand == 1:
        return f"{action_name} with right hand.".capitalize()
    if is_lhand == 1 and is_rhand == 1:
        return f"{action_name} with both hands.".capitalize()

def preprocessing_text():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    text_json = h2o_config.text_json
    
    text_description = {}
    for action in action_list[1:]:
        text_left = f"{action} with left hand.".capitalize()
        text_right = f"{action} with right hand.".capitalize()
        text_both = f"{action} with both hands.".capitalize()

        action_v, action_o = " ".join(action.split(" ")[:-1]), action.split(" ")[-1]
        action_ving = present_participle[action_v]
        text_left1 = f"{action_ving} {action_o} with left hand.".capitalize()
        text_right1 = f"{action_ving} {action_o} with right hand.".capitalize()
        text_both1 = f"{action_ving} {action_o} with both hands.".capitalize()

        action_3rd_v = third_verb[action_v]
        text_left2 = f"Left hand {action_3rd_v} {action_o}."
        text_right2 = f"Right hand {action_3rd_v} {action_o}."
        text_both2 = f"Both hands {action_v} {action_o}."

        action_passive = passive_verb[action_v]
        text_left3 = f"{action_o} {action_passive} with left hand.".capitalize()
        text_right3 = f"{action_o} {action_passive} with right hand.".capitalize()
        text_both3 = f"{action_o} {action_passive} with both hands.".capitalize()

        text_description[text_left] = [text_left, text_left1, text_left2, text_left3]
        text_description[text_right] = [text_right, text_right1, text_right2, text_right3]
        text_description[text_both] = [text_both, text_both1, text_both2, text_both3]

    with open(text_json, "w") as f:
        json.dump(text_description, f)

def preprocessing_balance_weights():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_path = h2o_config.data_path
    balance_weights_path = h2o_config.balance_weights_path
    t2c_json_path = h2o_config.t2c_json

    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]

    text_list = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        text_list.append(text_key)
    
    text_counter = Counter(text_list)
    text_dict = dict(text_counter)
    text_prob = {k:1/v for k, v in text_dict.items()}
    balance_weights = [text_prob[text] for text in text_list]
    with open(balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)
    with open(t2c_json_path, "w") as f:
        json.dump(text_dict, f)
        
def preprocessing_text2length():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_path = h2o_config.data_path
    t2l_json = h2o_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in text_dict:
            text_dict[text_key] = [num_frames]
        else:
            text_dict[text_key].append(num_frames)
    with open(t2l_json, "w") as f:
        json.dump(text_dict, f)

def print_text_data_num():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_path = h2o_config.data_path
    t2l_json_path = h2o_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")
