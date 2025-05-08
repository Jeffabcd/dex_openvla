import os
import os.path as osp

import numpy as np
import time
import tqdm
import glob
import json
import trimesh
import pickle
from collections import Counter

import torch
import tqdm
import cv2

from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.models.object_arctic import ObjectTensors
from lib.utils.data import processe_params, process_egocam_result
from lib.utils.file import load_config, read_json
from lib.utils.rot import axis_angle_to_rotmat
from lib.utils.frame import align_frame
from lib.utils.proc import (
    proc_torch_cuda, 
    proc_numpy, 
    transform_hand_to_xdata,
    transform_xdata_to_joints,  
    transform_obj_to_xdata, 
    farthest_point_sample, 
)
from lib.utils.proc_arctic import (
    process_hand_keypoints, 
    process_object, 
    get_contact_info_arctic, 
    process_text, 
    transform_joints_world_to_camera,
)
from constants.arctic_constants import (
    arctic_obj_name, 
    subj_list, 
    not_save_list, 
    action_list, 
    present_participle, 
    third_verb, 
    passive_verb, 
)

import h5py

def preprocessing_object():
    arctic_config = load_config("Text2HOI/configs/dataset/arctic.yaml")
    objects_folder = glob.glob(osp.join(arctic_config.obj_root, "*"))
   
    obj_pcs = {}
    obj_pc_normals = {}
    obj_pc_top = {}
    point_sets = {}
    obj_path = {}
    for object_folder in tqdm.tqdm(objects_folder):
        object_path = glob.glob(osp.join(object_folder, "mesh.obj"))[0]
        parts_path = glob.glob(osp.join(object_folder, "parts.json"))[0]
        parts = read_json(parts_path)
        parts = np.array(parts) # 0: top, 1: bottom
        mesh = trimesh.load(object_path, maintain_order=True)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()/1000
        sampled_normal = normal[0, point_set[0]].cpu().numpy()
        object_name = object_path.split("/")[-2]
        key = f"{object_name}"
        obj_pcs[key] = sampled_pc
        obj_pc_normals[key] = sampled_normal
        obj_pc_top[key] = 1-parts[point_set[0].cpu().numpy()] # 0: bottom, 1: top
        point_sets[key] = point_set[0].cpu().numpy()
        obj_path[key] = "/".join(object_path.split("/")[-2:])

    os.makedirs("data/arctic", exist_ok=True)
    with open("data/arctic/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": arctic_obj_name, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "obj_pc_top": obj_pc_top, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)
def preprocessing_new():
    start_time = time.time()
    arctic_config = load_config("Text2HOI/configs/dataset/arctic.yaml")
    data_root = arctic_config.root
    data_save_path = arctic_config.data_path
    text_root = arctic_config.text_root
    images_root = arctic_config.images_root

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=arctic_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=arctic_config.flat_hand)
    rhand_layer = rhand_layer.cuda()
    
    check_text_list = []

    total_lhand_joints = [] # #of data, #of frames, #of joints, 3
    total_rhand_joints = [] # #of data, #of frames, #of joints, 3
    total_instructions = []  # #of data, instruction
    total_images = [] # #of data, #of frames, 256, 256, 3
    total_sequence_names = []
    total_timestamps = []
    total_intrinsics = []

    world_frame = False
    print('subj_list', subj_list)
    print('arctic_obj_name', arctic_obj_name)
    
    for subj in subj_list:
        for object_name in tqdm.tqdm(arctic_obj_name, desc=f"{subj}"):
            hand_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.mano.npy"))
            hand_data_pathes.sort()
            obj_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.object.npy"))
            obj_data_pathes.sort()
            egocam_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.egocam.dist.npy"))
            egocam_data_pathes.sort()

            for hand_data_path, obj_data_path, egocam_data_path in zip(hand_data_pathes, obj_data_pathes, egocam_data_pathes):
                hand_data, object_data = processe_params(hand_data_path, obj_data_path)
                lhand_poses = hand_data["left.pose"]        
                lhand_betas = hand_data["left.shape"] # np.zeros((lhand_poses.shape[0], 10))
                lhand_trans = hand_data["left.trans"]
                lhand_joints = process_hand_keypoints(lhand_poses, lhand_betas, lhand_trans, lhand_layer)
                lhand_joints = proc_numpy(lhand_joints)
                rhand_poses = hand_data["right.pose"]
                rhand_betas = hand_data["right.shape"] # np.zeros((rhand_poses.shape[0], 10))
                rhand_trans = hand_data["right.trans"]
                rhand_joints = process_hand_keypoints(rhand_poses, rhand_betas, rhand_trans, rhand_layer)
                rhand_joints = proc_numpy(rhand_joints)
                
                if not world_frame:
                    SE3_matrices, intrinsics = process_egocam_result(egocam_data_path)

                    lhand_joints_cam = transform_joints_world_to_camera(lhand_joints, SE3_matrices)
                    rhand_joints_cam = transform_joints_world_to_camera(rhand_joints, SE3_matrices)
                else:
                    lhand_joints_cam = lhand_joints
                    rhand_joints_cam = rhand_joints

                data_basename = obj_data_path.split("/")[-1].replace(".object.npy", "")
                sequence_name = f'{subj}-{data_basename}'
                text_file = osp.join(text_root, subj, data_basename, "description.txt")
                with open(text_file, "r") as f:
                    text_data = f.readlines()


                duplicate_text = []
                for text_info in text_data:
                    if text_info == "\n":
                        continue
                    start_end_frame, action_name, hand_type, _ = text_info.split(" ")
                    if start_end_frame in duplicate_text:
                        continue
                    duplicate_text.append(start_end_frame)
                    
                    action_name = action_name.replace(",", "")
                    check_text = f"{action_name} {object_name}"
                    check_text_list.append(check_text)
                    if check_text in not_save_list:
                        continue
                    start_frame, end_frame = start_end_frame.split("-")
                    start_frame = int(start_frame)-1
                    end_frame = int(end_frame)+1-1
                    if end_frame-start_frame < 20:
                        continue

                    lhand_joints_list = lhand_joints_cam[start_frame:end_frame]
                    rhand_joints_list = rhand_joints_cam[start_frame:end_frame]
                    intrinsics_list = np.stack([intrinsics for _ in range(end_frame-start_frame)], axis=0)
                    #print('intrinsics_list.shape', intrinsics_list.shape)
                    print('len', end_frame-start_frame)
                                       # Load images for the current sequence
                    sequence_images = []
                    sequence_timestamps = []
                    for frame_idx in range(start_frame, end_frame):
                        # Path format: data/arctic/downloads/data/cropped_images/s01/box_grab_01/0/00003.jpg
                        img_path = osp.join(images_root, subj, data_basename, "0", f"{frame_idx+1:05d}.jpg")
                        sequence_timestamps.append(frame_idx)
                        if osp.exists(img_path):
                            # Load image using PIL and convert to numpy array
                            # from PIL import Image
                            # img = np.array(Image.open(img_path))
                            img = cv2.imread(img_path)
                            #('img', img.shape) (2000, 2800, 3)
                            h, w, c = img.shape
                            square_size = max(h, w)
                             # Calculate padding
                            top_pad = (square_size - h) // 2
                            bottom_pad = square_size - h - top_pad
                            left_pad = (square_size - w) // 2
                            right_pad = square_size - w - left_pad
                            #print('top_pad', top_pad, 'bottom_pad', bottom_pad, 'left_pad', left_pad, 'right_pad', right_pad)
                            square_image = cv2.copyMakeBorder(
                                img,
                                top=top_pad,
                                bottom=bottom_pad,
                                left=left_pad,
                                right=right_pad,
                                borderType=cv2.BORDER_REPLICATE
                            )
                            square_image = cv2.resize(square_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                            # print('img', square_image.shape)
                            # cv2.imwrite('test_image.jpg', square_image)
                            # raise
                            sequence_images.append(square_image)
                        else:
                            print(f"Warning: Image not found at {img_path}")
                    
                    if sequence_images:
                        total_images.append(np.array(sequence_images))
                    else:
                        print(f"Warning: No images found for {subj}/{data_basename} frames {start_frame+1}-{end_frame}")
                        continue


                    text_vis = f"{subj, data_basename, start_frame, end_frame}"
                    if hand_type == "both":
                        is_lhand = 1 
                        is_rhand = 1
                        instruction = f"{action_name} {object_name} with both hands.".capitalize()
                    elif hand_type == "left":
                        is_lhand = 1
                        is_rhand = 0
                        instruction = f"{action_name} {object_name} with left hand.".capitalize()
                    elif hand_type == "right":
                        is_rhand = 1
                        is_lhand = 0
                        instruction = f"{action_name} {object_name} with right hand.".capitalize()
                    else:
                        raise Exception("Hand type not supported")
                    
                    total_instructions.append(instruction)
                    total_sequence_names.append(sequence_name)
                    total_timestamps.append(np.array(sequence_timestamps))
                    total_lhand_joints.append(lhand_joints_list)
                    total_rhand_joints.append(rhand_joints_list)
                    total_intrinsics.append(intrinsics_list)
                    print('total_rhand_joints', np.array(rhand_joints_list).shape)
                    print('total_intrinsics', np.array(intrinsics_list).shape)

    fps = 5 
    ori_fps = 30

    for i in range(len(total_images)):
        fps_index = list(range(0, len(total_timestamps[i]), ori_fps//fps))
        total_images[i] = total_images[i][fps_index]
        total_lhand_joints[i] = total_lhand_joints[i][fps_index]
        total_rhand_joints[i] = total_rhand_joints[i][fps_index]
        total_timestamps[i] = total_timestamps[i][fps_index]
        total_intrinsics[i] = total_intrinsics[i][fps_index]
    print('total_images', len(total_images))
    print('total_instructions', len(total_instructions))
    print('total_lhand_joints', len(total_lhand_joints))
    print('total_rhand_joints', len(total_rhand_joints))
    print('total_sequence_names', len(total_sequence_names))
    print('total_timestamps', len(total_timestamps))
    print('total_intrinsics', len(total_intrinsics))
    # Create directory if it doesn't exist
    os.makedirs("data/arctic/preprocess_data", exist_ok=True)
    
    # Save data using h5py
    h5_path = "arctic/preprocess_data/arctic_data_intrinsics.h5"
    with h5py.File(h5_path, 'w') as f:
        # Create groups for different data types
        images_group = f.create_group('images')
        instructions_group = f.create_group('instructions')
        lhand_joints_group = f.create_group('lhand_joints')
        rhand_joints_group = f.create_group('rhand_joints')
        sequence_names_group = f.create_group('sequence_names')
        timestamps_group = f.create_group('timestamps')
        intrinsics_group = f.create_group('intrinsics')
        
        # Store each sequence
        for i in range(len(total_images)):
            # Store images as uint8 to save space
            images_group.create_dataset(f'seq_{i}', data=total_images[i], dtype='uint8', compression='gzip')
            
            # Store instructions as strings
            instructions_group.create_dataset(f'seq_{i}', data=np.array([total_instructions[i]], dtype='S'))
            
            # Store joint data
            lhand_joints_group.create_dataset(f'seq_{i}', data=total_lhand_joints[i], compression='gzip')
            rhand_joints_group.create_dataset(f'seq_{i}', data=total_rhand_joints[i], compression='gzip')
            
            # Store sequence names
            sequence_names_group.create_dataset(f'seq_{i}', data=np.array([total_sequence_names[i]], dtype='S'))
            
            # Store timestamps
            timestamps_group.create_dataset(f'seq_{i}', data=total_timestamps[i], compression='gzip')
            
            # Store intrinsics
            intrinsics_group.create_dataset(f'seq_{i}', data=total_intrinsics[i], compression='gzip')
        
        # Store metadata
        f.attrs['num_sequences'] = len(total_images)
        f.attrs['fps'] = fps
        f.attrs['original_fps'] = ori_fps
        # store original image size for usage of intrinsics
        f.attrs['ori_width'] = 2800
        f.attrs['ori_height'] = 2000
    
    print(f"Data successfully saved to {h5_path}")
    print("Finish:", time.time()-start_time)
 

def preprocessing_data():
    start_time = time.time()
    arctic_config = load_config("configs/dataset/arctic.yaml")
    data_root = arctic_config.root
    data_save_path = arctic_config.data_path
    text_root = arctic_config.text_root
    object_model = build_object_model(arctic_config.data_obj_pc_path)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=arctic_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=arctic_config.flat_hand)
    rhand_layer = rhand_layer.cuda()
    
    arctic_object_model = ObjectTensors("cuda")

    x_lhand_total = []
    x_rhand_total = []
    j_lhand_total = []
    j_rhand_total = []
    x_obj_total = []
    x_obj_angle_total = []
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
    object_name_total = []
    action_name_total = []
    nframes_total = []
    
    check_text_list = []

   
    print('subj_list', subj_list)
    print('arctic_obj_name', arctic_obj_name)
    
    for subj in subj_list:
        for object_name in tqdm.tqdm(arctic_obj_name, desc=f"{subj}"):
            hand_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.mano.npy"))
            hand_data_pathes.sort()
            obj_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.object.npy"))
            obj_data_pathes.sort()
            point_set, _, _, _, _   = object_model(object_name)
            for hand_data_path, obj_data_path in zip(hand_data_pathes, obj_data_pathes):
                hand_data, object_data = processe_params(hand_data_path, obj_data_path)
                lhand_poses = hand_data["left.pose"]
                print('poses', lhand_poses.shape)
                lhand_betas = hand_data["left.shape"] # np.zeros((lhand_poses.shape[0], 10))
                lhand_trans = hand_data["left.trans"]
                print('trans', lhand_trans.shape)
                lhand_joints = process_hand_keypoints(lhand_poses, lhand_betas, lhand_trans, lhand_layer)
                lhand_joints = proc_numpy(lhand_joints)
                print('joints', lhand_joints.shape)
                rhand_poses = hand_data["right.pose"]
                rhand_betas = hand_data["right.shape"] # np.zeros((rhand_poses.shape[0], 10))
                rhand_trans = hand_data["right.trans"]
                rhand_joints = process_hand_keypoints(rhand_poses, rhand_betas, rhand_trans, rhand_layer)
                rhand_joints = proc_numpy(rhand_joints)
                print('R joints', rhand_joints.shape)
                obj_angle = object_data["object.angle"]
                obj_rots = object_data["object.global_rot"]
                obj_trans = object_data["object.trans"]
                obj_dict = process_object(obj_angle, obj_rots, obj_trans/1000, object_name, arctic_object_model)
                obj_verts = obj_dict["v"]
                
                data_basename = obj_data_path.split("/")[-1].replace(".object.npy", "")
                text_file = osp.join(text_root, subj, data_basename, "description.txt")
                with open(text_file, "r") as f:
                    text_data = f.readlines()
                duplicate_text = []
                for text_info in text_data:
                    if text_info == "\n":
                        continue
                    start_end_frame, action_name, hand_type, _ = text_info.split(" ")
                    duplicate_text.append(f"{start_end_frame} {hand_type}")
                    if len(duplicate_text) != len(np.unique(duplicate_text)):
                        raise Exception("duplicate")
                    
                    action_name = action_name.replace(",", "")
                    check_text = f"{action_name} {object_name}"
                    check_text_list.append(check_text)
                    if check_text in not_save_list:
                        continue
                    start_frame, end_frame = start_end_frame.split("-")
                    start_frame = int(start_frame)-1
                    end_frame = int(end_frame)+1-1
                    if end_frame-start_frame < 20:
                        continue
                    lhand_pose_list = lhand_poses[start_frame:end_frame]
                    lhand_beta_list = lhand_betas[start_frame:end_frame]
                    lhand_trans_list = lhand_trans[start_frame:end_frame]
                    x_lhand_org_list = lhand_joints[start_frame:end_frame, 0]
                    rhand_pose_list = rhand_poses[start_frame:end_frame]
                    rhand_beta_list = rhand_betas[start_frame:end_frame]
                    rhand_trans_list = rhand_trans[start_frame:end_frame]
                    x_rhand_org_list = rhand_joints[start_frame:end_frame, 0]
                    print('x_lhand_org_list', x_lhand_org_list.shape)
                    obj_angle_list = obj_angle[start_frame:end_frame]
                    obj_rot_list = obj_rots[start_frame:end_frame]
                    obj_trans_list = obj_trans[start_frame:end_frame]
                    
                    obj_rot_list = proc_torch_cuda(obj_rot_list)
                    obj_trans_list = proc_torch_cuda(obj_trans_list)/1000 # mm -> m
                    obj_rotmat_list = axis_angle_to_rotmat(obj_rot_list)
                    obj_trans_list = obj_trans_list.unsqueeze(2)
                    obj_extmat_list = torch.cat([obj_rotmat_list, obj_trans_list], dim=2)
                    
                    lcf_idx, lcov_idx, lchj_idx, ldist_value, \
                    rcf_idx, rcov_idx, rchj_idx, rdist_value, \
                    is_lhand, is_rhand = get_contact_info_arctic(
                        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                        obj_verts[start_frame:end_frame, point_set], 
                        lhand_layer, rhand_layer, 
                    )
                    assert is_lhand != 0 or is_rhand != 0
                    
                    text_vis = f"{subj, data_basename, start_frame, end_frame}"
                    if hand_type == "both":
                        assert is_lhand == 1 and is_rhand == 1, text_vis
                    elif hand_type == "left":
                        assert is_lhand == 1, text_vis
                        is_rhand = 0
                    elif hand_type == "right":
                        assert is_rhand == 1, text_vis
                        is_lhand = 0
                    else:
                        raise Exception("Hand type not supported")
                    
                    x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
                    x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
                    j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                    j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                    print('j_lhand', j_lhand.shape)

                    j_lhand_total.append(j_lhand)
                    j_rhand_total.append(j_rhand)
                    x_obj = transform_obj_to_xdata(obj_extmat_list)
                    x_obj_angle = obj_angle_list
                    x_lhand_total.append(x_lhand)
                    x_rhand_total.append(x_rhand)
                    x_obj_total.append(x_obj)
                    x_obj_angle_total.append(x_obj_angle)
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
                    object_name_total.append(object_name)
                    action_name_total.append(action_name)
                    nframes_total.append(len(obj_extmat_list))

    total_dict = {
        "x_lhand": x_lhand_total,
        "x_rhand": x_rhand_total,
        "j_lhand": j_lhand_total,
        "j_rhand": j_rhand_total, 
        "x_obj": x_obj_total,
        "x_obj_angle": x_obj_angle_total,
        "lhand_beta": lhand_beta_total,
        "rhand_beta": rhand_beta_total,
        "lhand_org": x_lhand_org_total, 
        "rhand_org": x_rhand_org_total, 
    }
    final_dict = align_frame(total_dict)
    
    np.savez(
        data_save_path, 
        **final_dict, 
        lcf_idx=np.array(lcf_idx_total), 
        lcov_idx=np.array(lcov_idx_total), 
        lchj_idx=np.array(lchj_idx_total), 
        ldist_value=np.array(ldist_value_total), 
        rcf_idx=np.array(rcf_idx_total), 
        rcov_idx=np.array(rcov_idx_total), 
        rchj_idx=np.array(rchj_idx_total), 
        rdist_value=np.array(rdist_value_total), 
        is_lhand=np.array(is_lhand_total), 
        is_rhand=np.array(is_rhand_total), 
        object_name=np.array(object_name_total),
        action_name=np.array(action_name_total),
        nframes=np.array(nframes_total),
    )
    print("Finish:", time.time()-start_time)
    print("Length of data:", len(x_obj_total))
    
def preprocessing_text():
    text_description = {}
    for action in action_list:
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

    with open("data/arctic/text.json", "w") as f:
        json.dump(text_description, f)
        
def preprocessing_balance_weights(is_action=False):
    arctic_config = load_config("configs/dataset/arctic.yaml")
    if is_action:
        data_path = arctic_config.action_train_data_path
        balance_weights_path = arctic_config.action_balance_weights_path
    else:
        data_path = arctic_config.data_path
        balance_weights_path = arctic_config.balance_weights_path
    t2c_json_path = arctic_config.t2c_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        object_name = data["object_name"]

    text_list = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], 
            object_name[i], 
            is_lhand[i], is_rhand[i], 
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
    with np.load("data/arctic/data.npz", allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        object_name = data["object_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], 
            object_name[i], 
            is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in text_dict:
            text_dict[text_key] = [num_frames]
        else:
            text_dict[text_key].append(num_frames)
    with open("data/arctic/text_length.json", "w") as f:
        json.dump(text_dict, f)

def print_text_data_num():
    arctic_config = load_config("configs/dataset/arctic.yaml")
    data_path = arctic_config.data_path
    t2l_json_path = arctic_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")