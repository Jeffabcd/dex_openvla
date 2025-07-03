import os
import os.path as osp

import numpy as np
import time
import tqdm
import glob
import json
# import trimesh
import pickle
from collections import Counter

import torch
import tqdm



import h5py

from moviepy import VideoFileClip
from moviepy import ImageSequenceClip
import torch.nn as nn
import pandas as pd
import re
import cv2
import yaml

from easydict import EasyDict as edict

import sys
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))

import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def visuliaze_joints(fps, j_lhand, j_rhand, image_list, intrinsic_matrix, resolution, action_name):
    """
    Visualize hand joints on images and create a video
    Args:
        j_lhand, j_rhand: hand joints in camera coordinates (T, 25, 3)
        image_list: list of images
        intrinsic_matrix: 3x3 camera intrinsic matrix
        resolution: original resolution (width, height)
    """
    print('resolution', resolution)
    print('image_list', len(image_list))
    print('j_lhand', j_lhand.shape)
    # Get current image size
    current_h, current_w = image_list[0].shape[:2]
    
    # Scale intrinsic matrix according to image resize
    scale_w = current_w / resolution[0]
    scale_h = current_h / resolution[1]
    # print('scale_w', scale_w)
    # print('scale_h', scale_h)
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
            if np.any(np.isnan(joint)):
                continue
            cv2.circle(img_draw, (int(joint[0]), int(joint[1])), 5, (0,0,255), -1)
        for conn in connections:
            pt1, pt2 = j_l_2d[conn[0]], j_l_2d[conn[1]]
            if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                continue
            cv2.line(img_draw, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    (0,0,255), 1)
            
        # Right hand (blue)
        for joint in j_r_2d:
            if np.any(np.isnan(joint)):
                continue
            cv2.circle(img_draw, (int(joint[0]), int(joint[1])), 5, (255,0,0), -1)
        for conn in connections:
            pt1, pt2 = j_r_2d[conn[0]], j_r_2d[conn[1]]
            if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                continue
            cv2.line(img_draw, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    (255,0,0), 1)
        
        cv2.putText(img_draw, action_name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        frames.append(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    
    # Create video using moviepy
    from moviepy import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile("hand_joints_visualization.mp4")
    
    return frames

def load_video_moviepy(video_path):
    """
    Load a video file into a numpy array using moviepy.
    
    Args:
        video_path (str): Path to the .mp4 file
        
    Returns:
        frames (np.array): Array of frames with shape (num_frames, height, width, channels)
        fps (float): Frames per second of the video
    """
    # Load the video
    clip = VideoFileClip(video_path)
    
    # Get the frames
    frames = []
    for frame in clip.iter_frames():
        frames.append(frame)
    
    # Convert to numpy array
    frames = np.array(frames)
    
    # print(f"Loaded video with {len(frames)} frames")
    # print(f"Frame shape: {frames[0].shape}")
    # print(f"FPS: {clip.fps}")
    
    # Close the clip to free resources
    clip.close()
    
    return frames, clip.fps

def find_valid_sequences(total_frames, nan_frame_indices, min_length=75):
    """
    Find valid continuous sequences between NaN frames that meet minimum length requirement.
    
    Args:
        total_frames (int): Total number of frames in sequence
        nan_frame_indices (np.ndarray): Array of frame indices containing NaN
        min_length (int): Minimum required length for a valid sequence
        
    Returns:
        list of tuples: Each tuple contains (start_idx, end_idx) for valid sequences
    """
    # Add sequence boundaries to nan_frames
    nan_frames = set(nan_frame_indices)
    nan_frames.add(-1)  # Add virtual frame before start
    nan_frames.add(total_frames)  # Add virtual frame after end
    
    # Sort all break points
    break_points = sorted(list(nan_frames))
    
    # Find valid sequences
    valid_sequences = []
    for i in range(len(break_points) - 1):
        start = break_points[i] + 1
        end = break_points[i + 1]
        
        # Check if sequence length meets minimum requirement
        if end - start >= min_length:
            valid_sequences.append((start, end))
    
    return valid_sequences

def find_nan_in_hand_pose(hand_pose):
    # Find frames that contain any NaN values
    nan_frames = np.any(np.isnan(hand_pose), axis=(1,2))
    nan_frame_indices = np.where(nan_frames)[0]
    
    if len(nan_frame_indices) > 0:
        print(f'Found {len(nan_frame_indices)} frames with NaN values')
        print(f'Frames with NaN: {nan_frame_indices}')
        
        # Find valid sequences
        valid_sequences = find_valid_sequences(len(hand_pose), nan_frame_indices)
        if valid_sequences:
            print('Valid sequences (start, end):')
            for start, end in valid_sequences:
                print(f'  {start} to {end} (length: {end-start})')
        else:
            print('No valid sequences found meeting minimum length requirement')
        
    else:
        print('No NaN values found in hand pose data')
        valid_sequences = [(0, len(hand_pose))]
        
    return nan_frame_indices, valid_sequences


def get_hand_pose(hawor_args, is_lhand, is_rhand):
    fx, fy = 600, 600
    cx, cy = 960, 540

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(hawor_args)
    print(f'start: {start_idx}, end_idx: {end_idx}')

    frame_chunks_all, img_focal = hawor_motion_estimation(hawor_args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path) or True:
        calib = hawor_slam(hawor_args, start_idx, end_idx) # 4
        cx = calib[2]
        cy = calib[3]
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(hawor_args, start_idx, end_idx, frame_chunks_all)
    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }

    total_right_verts = []
    total_left_verts = []

    right_valid_sequences = []
    left_valid_sequences = []

    right_verts = None
    left_verts = None

    R_w2c_p = R_w2c_sla_all.cpu().numpy()
    # t_w2c_p = t_w2c_sla_all.cpu().numpy()

    nan_frames = np.any(np.isnan(R_w2c_p), axis=(1,2))
    nan_frame_indices = np.where(nan_frames)[0]
    
    if len(nan_frame_indices) > 0:
        print('='*40)
        print('R_w2c_p has NaN values')
        print(f'Found {len(nan_frame_indices)} frames with NaN values')
        print(f'Frames with NaN: {nan_frame_indices}')
        

    if is_rhand:
        # get right hand vertices
        hand = 'right'
        hand_idx = hand2idx[hand]
        _v_start = 0
        _v_end = pred_trans.shape[1]

        pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, _v_start:_v_end], pred_rot[hand_idx:hand_idx+1, _v_start:_v_end], pred_hand_pose[hand_idx:hand_idx+1, _v_start:_v_end], betas=pred_betas[hand_idx:hand_idx+1, _v_start:_v_end])
        right_verts = pred_glob_r['joints'][0]

        if isinstance(right_verts, torch.Tensor):
            right_verts = right_verts.cpu().numpy()


        nan_idx, valid_sequences = find_nan_in_hand_pose(right_verts)

        for i in range(len(right_verts)):
            frame_idx = i
            R_w2c = R_w2c_sla_all[i].cpu().numpy()
            t_w2c = t_w2c_sla_all[i].cpu().numpy()
            right_verts_cam = np.dot(R_w2c, right_verts[frame_idx].T) + t_w2c.reshape(3, 1)
            right_verts[i] = right_verts_cam.T

        
        # print('right_verts', right_verts.shape)

    if is_lhand:
        # get left hand vertices
        hand = 'left'
        hand_idx = hand2idx[hand]
        pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
        left_verts = pred_glob_l['joints'][0]
        if isinstance(left_verts, torch.Tensor):
            left_verts = left_verts.cpu().numpy()

        l_nan_idx, left_valid_sequences = find_nan_in_hand_pose(left_verts)

        for i in range(len(left_verts)):
            frame_idx = i
            R_w2c = R_w2c_sla_all[i].cpu().numpy()
            t_w2c = t_w2c_sla_all[i].cpu().numpy()
            left_verts_cam = np.dot(R_w2c, left_verts[frame_idx].T) + t_w2c.reshape(3, 1)
            left_verts[i] = left_verts_cam.T

        if is_rhand:
            nan_idx_s = set(nan_idx)
            nan_idx_s.update(l_nan_idx)
            if len(nan_idx_s) > 0:
                # contains nan 
                valid_sequences = find_valid_sequences(len(right_verts), list(nan_idx_s))
        else:
            valid_sequences = left_valid_sequences
        

    return right_verts, left_verts, valid_sequences, intrinsic_matrix

def load_cmd_tuple_hdf5(path, img_key="observation.image.left"):

    with h5py.File(path, 'r') as file:
        # Processed HDF5
        # Show two potential options
        assert "observation.image.left" in file
        assert "observation.image.right" in file
        
        # Get images
        compresses_img_arr = file[img_key][()]
        img_list = []
        for i in range(compresses_img_arr.shape[0]):
            cur_img = cv2.imdecode(compresses_img_arr[i], cv2.IMREAD_COLOR)
            img_list.append(cur_img)

    return img_list

def get_task_name(obj):
    # Extract category number and task number from path
    task_type = obj['task_type']
    object_name = obj['object']
    is_lhand = obj['left_hand']
    is_rhand = obj['right_hand']

    task_name = f"{task_type} {object_name}"

    if is_lhand and is_rhand:
       instruction = f"{task_name} with both hands.".capitalize()
    elif is_lhand:
       instruction = f"{task_name} with left hand.".capitalize()
    elif is_rhand:
       instruction = f"{task_name} with right hand.".capitalize()

    return instruction
        
    


def preprocessing_new():
    start_time = time.time()
    humanpolicy_config = load_config("humanpolicy.yaml")
    root = humanpolicy_config.root
    data_save_path = humanpolicy_config.data_path
    metadata_path = humanpolicy_config.metadata_path
    mano_root = humanpolicy_config.mano_root
    
    hawor_args = edict()
    hawor_args.checkpoint = "weights/hawor/checkpoints/hawor.ckpt"
    hawor_args.infiller_weight = "weights/hawor/checkpoints/infiller.pt"
    hawor_args.video_path = ''
    hawor_args.input_type = "file"
    hawor_args.vis_mode = "world"
    hawor_args.img_focal = 600
    # 1 740, 2 678 4 677
    # ./ZY2021080000*/H*/C*/N*/S*/s*/T*/

    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    # Get per_task_attributes dictionary
    per_task_attrs = data.get('per_task_attributes', {})
    
    # Filter objects where embodiment_type contains 'human'
    human_tasks = {
        task_id: attrs 
        for task_id, attrs in per_task_attrs.items()
        if 'human' in attrs.get('embodiment_type', '').lower()
    }

    print(len(human_tasks))
    # raise Exception("stop")


    total_count = 0
    valid_count = 0
    step = 10
    count = 0
    for task_id, attrs in human_tasks.items():
        check_text_list = []

        total_lhand_joints = [] # #of data, #of frames, #of joints, 3
        total_rhand_joints = [] # #of data, #of frames, #of joints, 3
        total_instructions = []  # #of data, instruction
        total_images = [] # #of data, #of frames, 256, 256, 3
        total_sequence_names = []
        total_timestamps = []
        total_intrinsic_matrix = []
        total_resolution = []

        h5_path = f"data/humanpolicy/preprocess_data/humanpolicy_data_{task_id}.h5"
        if os.path.exists(h5_path):
            continue

        data_list = glob.glob(osp.join(root,task_id, f"*.hdf5"))
        if len(data_list) == 0:
            continue
            
            #raise Exception(f"No data found for {task_id}")

        video_dir = osp.join(root,task_id, 'video')
        os.makedirs(video_dir, exist_ok=True)
        
        instruction = get_task_name(attrs)
        is_lhand = attrs['left_hand'] 
        is_rhand = attrs['right_hand']

        for data_path in tqdm.tqdm(data_list):
            print(f'total_count / valid_count: {total_count} / {valid_count}')

            sequence_name = data_path.split("/")[-2:]
            _data_name = data_path.split("/")[-1].split(".")[0]
            sequence_name = "/".join(sequence_name)
            frames = load_cmd_tuple_hdf5(data_path)

            video_path = osp.join(video_dir, f'{_data_name}.mp4')
            if not os.path.exists(video_path):
                clip = ImageSequenceClip(frames, fps=30)
                clip.write_videofile(video_path, codec='libx264')

            ori_fps = 30
            hawor_args.video_path = video_path


            # try:
            #     rhand_pose, lhand_pose, intrinsic_matrix = get_hand_pose(hawor_args, is_lhand, is_rhand)
            # except:
            #     print(image_path, 'failed!')
            #     continue
            total_count += 1
            try:
            
                right_verts, left_verts, valid_sequences, intrinsic_matrix = get_hand_pose(hawor_args, is_lhand, is_rhand)

            except Exception as e:
                print(video_path, 'failed! _ slam')
                print(e)
                continue
            valid_count += 1

            if not is_lhand and not is_rhand:
                print('='*40)
                print(video_path, 'failed! _ data')
                continue
            if len(valid_sequences) == 0:
                print('='*40)
                print(video_path, 'failed! _ hawor')
                continue

            total_frames = []
            total_ts = []
            total_s_name = []
            total_in = []
            total_re = []
            total_inst = []
            total_right_verts = []
            total_left_verts = []
            
            for k , (vis_start, vis_end) in enumerate(valid_sequences):
                right_verts_list = [] if right_verts is None else right_verts[vis_start:vis_end]
                left_verts_list = [] if left_verts is None else left_verts[vis_start:vis_end]
                

                if right_verts is not None:
                    total_right_verts.append(np.array(right_verts_list))
                    if left_verts is None:
                        total_left_verts.append(np.zeros(total_right_verts[-1].shape))
                if left_verts is not None:
                    total_left_verts.append(np.array(left_verts_list))
                    if right_verts is None:
                        total_right_verts.append(np.zeros(total_left_verts[-1].shape))

                frames_resized = []
                for frame in frames[vis_start: vis_end]:
                    frame_resized = np.array(Image.fromarray(frame).resize((256, 256), Image.LANCZOS))
                    frames_resized.append(frame_resized)

                total_frames.append(np.array(frames_resized))
                total_ts.append(np.arange(vis_start, vis_end))
                total_s_name.append(sequence_name)
                total_in.append(intrinsic_matrix)
                total_re.append((frames[0].shape[1], frames[0].shape[0]))
                total_inst.append(instruction)
            
                # visuliaze_joints(ori_fps, total_left_verts[-1], total_right_verts[-1], frames[vis_start:vis_end], intrinsic_matrix, (frames[0].shape[1], frames[0].shape[0]), instruction)
                # raise Exception("stop")

            
            # raise
            total_lhand_joints.extend(total_left_verts)
            total_rhand_joints.extend(total_right_verts)
            total_instructions.extend(total_inst)

            total_images.extend(total_frames)
            total_sequence_names.extend(total_s_name)
            total_timestamps.extend(total_ts)
            total_intrinsic_matrix.extend(total_in)
            total_resolution.extend(total_re)

            print(sequence_name)
            count += len(total_inst)
            
            # raise Exception("stop")  
        
        print('total_images', len(total_images))
        print('total_instructions', len(total_instructions))
        print('total_lhand_joints', len(total_lhand_joints))
        print('total_rhand_joints', len(total_rhand_joints))
        print('total_sequence_names', len(total_sequence_names))
        print('total_timestamps', len(total_timestamps))

        fps = 5 
        

        
        for i in range(len(total_images)):
            fps_index = list(range(0, len(total_timestamps[i]), int(ori_fps)//fps))
            print('lll', len(total_timestamps[i]))
            #print('fps index', fps_index)
            print('kl;lklkl', total_images[i].shape)
            total_images[i] = total_images[i][fps_index]
            total_lhand_joints[i] = total_lhand_joints[i][fps_index]
            total_rhand_joints[i] = total_rhand_joints[i][fps_index]
            total_timestamps[i] = total_timestamps[i][fps_index]


        # Create directory if it doesn't exist
        os.makedirs("data/HOI4D/preprocess_data", exist_ok=True)
        
        # Save data using h5py
        with h5py.File(h5_path, 'w') as f:
            # Create groups for different data types
            images_group = f.create_group('images')
            instructions_group = f.create_group('instructions')
            lhand_joints_group = f.create_group('lhand_joints')
            rhand_joints_group = f.create_group('rhand_joints')
            sequence_names_group = f.create_group('sequence_names')
            timestamps_group = f.create_group('timestamps')
            intrinsics_group = f.create_group('intrinsics')
            resolutions_group = f.create_group('resolutions')
            
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
                intrinsics_group.create_dataset(f'seq_{i}', data=total_intrinsic_matrix[i], compression='gzip')

                # Store resolutions
                resolutions_group.create_dataset(f'seq_{i}', data=total_resolution[i], compression='gzip')
            
            # Store metadata
            f.attrs['num_sequences'] = len(total_images)
            f.attrs['fps'] = fps
            f.attrs['original_fps'] = ori_fps
        
        print(f"Data successfully saved to {h5_path}")
        print("Finish:", time.time()-start_time)
    print(f'total sequence: {count}')
 

if __name__ == "__main__":
    preprocessing_new()
