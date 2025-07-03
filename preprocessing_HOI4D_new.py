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
        
        cv2.putText(img_draw, action_name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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


        nan_idx, right_valid_sequences = find_nan_in_hand_pose(right_verts)

        for vis_start, vis_end in right_valid_sequences:
            verts_list = []
            for i in range(vis_start, vis_end):
                R_w2c = R_w2c_sla_all[i].cpu().numpy()
                t_w2c = t_w2c_sla_all[i].cpu().numpy()
                
                right_verts_cam = np.dot(R_w2c, right_verts[i].T) + t_w2c.reshape(3, 1)
                verts_list.append(right_verts_cam.T)
            total_right_verts.append(np.array(verts_list))
        # print('right_verts', right_verts.shape)

    if is_lhand:
        # get left hand vertices
        hand = 'left'
        hand_idx = hand2idx[hand]
        pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
        left_verts = pred_glob_l['joints'][0]
        if isinstance(left_verts, torch.Tensor):
            left_verts = left_verts.cpu().numpy()

        nan_idx, left_valid_sequences = find_nan_in_hand_pose(left_verts)
        for vis_start, vis_end in left_valid_sequences:
            verts_list = []
            for i in range(vis_start, vis_end):
                R_w2c = R_w2c_sla_all[i].cpu().numpy()
                t_w2c = t_w2c_sla_all[i].cpu().numpy()
                
                left_verts_cam = np.dot(R_w2c, left_verts[i].T) + t_w2c.reshape(3, 1)
                verts_list.append(left_verts_cam.T)
            total_left_verts.append(np.array(verts_list))

    return total_right_verts, total_left_verts, right_valid_sequences, left_valid_sequences, intrinsic_matrix



def get_task_name(df, sequence_name, is_lhand, is_rhand):
    # Extract category number and task number from path
    category_match = re.search(r'C(\d+)', sequence_name)
    task_match = re.search(r'T(\d+)', sequence_name)

    category_num = int(category_match.group(1))
    task_num = int(task_match.group(1))
    
    # Find the corresponding row in the CSV
    category_row = df[df['Category ID'] == f'C{category_num}']
    
    if category_row.empty:
        raise ValueError(f"Category C{category_num} not found in task definitions")
    
    # Get the task name from the corresponding T column
    task_col = f'T{task_num}'
    if task_col not in df.columns:
        raise ValueError(f"Task column {task_col} not found in task definitions")
    
    task_name = category_row[task_col].iloc[0]
    
    if pd.isna(task_name) or task_name == 'N/A':
        raise ValueError(f"No task defined for C{category_num} {task_col}")
    
    if is_rhand:
        instruction = f"{task_name} with right hand.".capitalize()
    else:
        instruction = f"{task_name} with left hand.".capitalize()
    #if is_lhand and is_rhand:
    #    instruction = f"{task_name} with both hands.".capitalize()
    #elif is_lhand:
    #    instruction = f"{task_name} with left hand.".capitalize()
    #elif is_rhand:
    #    instruction = f"{task_name} with right hand.".capitalize()

    return instruction
        
    


def preprocessing_new():
    start_time = time.time()
    HOI4D_config = load_config("HOI4D.yaml")
    image_root = HOI4D_config.image_root
    hand_root = HOI4D_config.hand_root
    camera_root = HOI4D_config.camera_root
    data_save_path = HOI4D_config.data_path
    text_path = HOI4D_config.text_path
    mano_root = HOI4D_config.mano_root
    
    hawor_args = edict()
    hawor_args.checkpoint = "weights/hawor/checkpoints/hawor.ckpt"
    hawor_args.infiller_weight = "weights/hawor/checkpoints/infiller.pt"
    hawor_args.video_path = ''
    hawor_args.input_type = "file"
    hawor_args.vis_mode = "world"
    hawor_args.img_focal = 600
    # 1 754, 2 684  3 421 4 
    camera_num = 4
    # ./ZY2021080000*/H*/C*/N*/S*/s*/T*/

    image_paths = glob.glob(osp.join(image_root, f"ZY2021080000{camera_num}", "H*", "C*", "N*", "S*", "s*", "T*",'align_rgb','image.mp4'))
    image_paths.sort()

    camera_intrinsic = np.load(osp.join(camera_root, f"ZY2021080000{camera_num}",'intrin.npy'))

    print('camera_intrinsic', camera_intrinsic)

    df = pd.read_csv(text_path)
    
    step = 10
    count = 0
    for start_idx in range(0, len(image_paths), step):
        check_text_list = []

        total_lhand_joints = [] # #of data, #of frames, #of joints, 3
        total_rhand_joints = [] # #of data, #of frames, #of joints, 3
        total_instructions = []  # #of data, instruction
        total_images = [] # #of data, #of frames, 256, 256, 3
        total_sequence_names = []
        total_timestamps = []
        total_intrinsic_matrix = []
        total_resolution = []

        end_idx = min(start_idx+step, len(image_paths))
        h5_path = f"data/HOI4D/preprocess_data/HOI4D_data_{camera_num}_{start_idx}_{end_idx}.h5"
        if os.path.exists(h5_path):
            continue

        for image_path in tqdm.tqdm(image_paths[start_idx:end_idx]):
            sequence_name = image_path.split("/")[-9:-2]
            sequence_name = "/".join(sequence_name)

            # check hand pose
            left_hand_path = osp.join(hand_root,'handpose_left_hand', sequence_name)
            is_lhand = osp.exists(left_hand_path)
            right_hand_path = osp.join(hand_root,'handpose_right_hand', sequence_name)
            is_rhand = osp.exists(right_hand_path)
            
            if not is_lhand and not is_rhand:
                print('='*40)
                print(image_path, 'failed! _ data')
                continue
            
            frames, ori_fps = load_video_moviepy(image_path)
            hawor_args.video_path = image_path
            # try:
            #     rhand_pose, lhand_pose, intrinsic_matrix = get_hand_pose(hawor_args, is_lhand, is_rhand)
            # except:
            #     print(image_path, 'failed!')
            #     continue

            try:
            
                total_right_verts, total_left_verts, right_valid_sequences, left_valid_sequences, intrinsic_matrix = get_hand_pose(hawor_args, is_lhand, is_rhand)

            except:
                print(image_path, 'failed! _ slam')
                continue

            if len(total_right_verts) == 0 and len(total_left_verts) == 0:
                print('='*40)
                print(image_path, 'failed! _ hawor')
                continue


            instruction = get_task_name(df, sequence_name, is_lhand, is_rhand)

            total_frames = []
            total_ts = []
            total_s_name = []
            total_in = []
            total_re = []
            total_inst = []
            if total_right_verts:
                total_left_verts = []
                for k , (vis_start, vis_end) in enumerate(right_valid_sequences):
                    total_right_verts[k] = proc_numpy(total_right_verts[k])
                    total_left_verts.append(np.zeros(total_right_verts[k].shape))
                    frames_resized = []
                    for frame in frames[vis_start: vis_end]:
                        frame_resized = np.array(Image.fromarray(frame).resize((256, 256), Image.LANCZOS))
                        frames_resized.append(frame_resized)

                    #print('8888', np.array(frames_resized).shape)
                    total_frames.append(np.array(frames_resized))
                    total_ts.append(np.arange(vis_start, vis_end))
                    total_s_name.append(sequence_name)
                    total_in.append(intrinsic_matrix)
                    total_re.append((frames[0].shape[1], frames[0].shape[0]))
                    total_inst.append(instruction)
            else:
                total_right_verts = []
                for k , (vis_start, vis_end) in enumerate(left_valid_sequences):
                    total_left_verts[k] = proc_numpy(total_left_verts[k])
                    total_right_verts.append(np.zeros(total_left_verts[k].shape))
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
                
            # raise
            # visuliaze_joints(ori_fps, lhand_pose, rhand_pose, frames, intrinsic_matrix, (frames[0].shape[1], frames[0].shape[0]), instruction)
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
            print('fps index', fps_index)
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

class HOI4Ddataset():
    def __init__(self, camera_num=4):
        HOI4D_config = load_config("HOI4D.yaml")
        self.image_root = HOI4D_config.image_root
        self.data_save_path = HOI4D_config.data_path
        self.camera_num = camera_num
        
        # Load all H5 files for this camera
        self.h5_files = []
        self.sequence_info = {}  # Store {new_sequence_name: (h5_file_path, seq_idx)}
        
        h5_pattern = f"data/HOI4D/preprocess_data/HOI4D_data_*.h5"
        h5_paths = glob.glob(h5_pattern)
        h5_paths.sort()
        
        for h5_path in h5_paths:
            h5_file_name = os.path.basename(h5_path).replace('.h5', '')
            with h5py.File(h5_path, 'r') as f:
                num_sequences = f.attrs['num_sequences']
                for seq_idx in range(num_sequences):
                    sequence_name = f['sequence_names'][f'seq_{seq_idx}'][0].decode('utf-8')
                    timestamps = f['timestamps'][f'seq_{seq_idx}'][:]
                    start_frame = timestamps[0]
                    end_frame = timestamps[-1]
                    # Create unique identifier with timestamp range
                    new_sequence_name = f"{h5_file_name}++{sequence_name}++{start_frame}_{end_frame}"
                    self.sequence_info[new_sequence_name] = (h5_path, seq_idx)
        
        print(f"Loaded {len(self.sequence_info)} sequences from {len(h5_paths)} H5 files")
        
        # Current sequence data
        self.current_h5_file = None
        self.current_h5_data = None
        self.current_seq_idx = None
        self.sequence_name = None
        self.new_sequence_name = None
        self.intrinsic_matrix = None
        self.resolution = None
        self.lhand_joints_cam = None
        self.rhand_joints_cam = None
        self.timestamps = None
        self.instruction = None
        self.original_video_frames = None
        self.original_fps = None

    def set_sequence(self, new_sequence_name):
        """Set the current sequence by new_sequence_name (h5_file_name++sequence_name++start_end)"""
        if new_sequence_name not in self.sequence_info:
            raise ValueError(f"Sequence '{new_sequence_name}' not found")
        
        h5_path, seq_idx = self.sequence_info[new_sequence_name]
        
        # Close previous H5 file if different
        if self.current_h5_file != h5_path:
            if self.current_h5_data is not None:
                self.current_h5_data.close()
            self.current_h5_data = h5py.File(h5_path, 'r')
            self.current_h5_file = h5_path
        
        self.current_seq_idx = seq_idx
        self.new_sequence_name = new_sequence_name
        
        # Load sequence data from H5
        self.sequence_name = self.current_h5_data['sequence_names'][f'seq_{seq_idx}'][0].decode('utf-8')
        self.intrinsic_matrix = self.current_h5_data['intrinsics'][f'seq_{seq_idx}'][:]
        self.resolution = tuple(self.current_h5_data['resolutions'][f'seq_{seq_idx}'][:])
        self.lhand_joints_cam = self.current_h5_data['lhand_joints'][f'seq_{seq_idx}'][:]
        self.rhand_joints_cam = self.current_h5_data['rhand_joints'][f'seq_{seq_idx}'][:]
        self.timestamps = self.current_h5_data['timestamps'][f'seq_{seq_idx}'][:]
        self.instruction = self.current_h5_data['instructions'][f'seq_{seq_idx}'][0].decode('utf-8')
        
        # Load original video frames (only once per video, not per sequence)
        video_path = osp.join(self.image_root, self.sequence_name, 'align_rgb', 'image.mp4')
        
        if not osp.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Only load video if it's different from current one
        if self.original_video_frames is None or self.current_video_path != video_path:
            self.original_video_frames, self.original_fps = load_video_moviepy(video_path)
            self.current_video_path = video_path
        
        print(f"Set sequence: {new_sequence_name}")
        print(f"Original sequence: {self.sequence_name}")
        print(f"Frames: {len(self.timestamps)} (from {self.timestamps[0]} to {self.timestamps[-1]})")
        print(f"Instruction: {self.instruction}")

    def get_sequences_by_video(self, sequence_name):
        """Get all sequences from the same original video"""
        matching_sequences = []
        for new_seq_name in self.sequence_info.keys():
            if sequence_name in new_seq_name:
                matching_sequences.append(new_seq_name)
        return sorted(matching_sequences)

    def get_num_sequences(self):
        """Get total number of sequences"""
        return len(self.sequence_info)

    def get_num_frames(self):
        """Get number of frames in current sequence"""
        if self.timestamps is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        return len(self.timestamps)

    def get_3d_joints(self, timestamp):
        """Get 3D joints for left and right hands at given timestamp"""
        if self.lhand_joints_cam is None or self.rhand_joints_cam is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        
        if timestamp >= len(self.lhand_joints_cam):
            raise ValueError(f"Timestamp {timestamp} out of range (0-{len(self.lhand_joints_cam)-1})")
        
        j_lhand = self.lhand_joints_cam[timestamp]  # (21, 3)
        j_rhand = self.rhand_joints_cam[timestamp]  # (21, 3)
        
        return j_lhand, j_rhand

    def get_2d_joints(self, timestamp):
        """Get 2D joints for left and right hands at given timestamp"""
        j_lhand, j_rhand = self.get_3d_joints(timestamp)
        
        # Get current image dimensions from original video
        original_timestamp = self.timestamps[timestamp]
        current_img = self.original_video_frames[original_timestamp]
        current_h, current_w = current_img.shape[:2]
        
        # Scale intrinsic matrix according to image resize
        scale_w = current_w / self.resolution[0]
        scale_h = current_h / self.resolution[1]
        scaled_intrinsic = self.intrinsic_matrix.copy()
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy
        
        # Project 3D points to 2D
        j_l_2d = (scaled_intrinsic @ j_lhand.T).T  # (21, 3)
        j_l_2d = j_l_2d[:, :2] / j_l_2d[:, 2:]  # Perspective division
        j_r_2d = (scaled_intrinsic @ j_rhand.T).T
        j_r_2d = j_r_2d[:, :2] / j_r_2d[:, 2:]
        
        return j_l_2d, j_r_2d

    def get_image(self, timestamp):
        """Get original resolution image from the dataset directory"""
        if self.original_video_frames is None or self.timestamps is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        
        if timestamp >= len(self.timestamps):
            raise ValueError(f"Timestamp {timestamp} out of range (0-{len(self.timestamps)-1})")
        
        # Get original timestamp and return corresponding frame
        original_timestamp = self.timestamps[timestamp]
        
        if original_timestamp >= len(self.original_video_frames):
            raise ValueError(f"Original timestamp {original_timestamp} out of range for video")
        
        return self.original_video_frames[original_timestamp]

    def get_instruction(self):
        """Get instruction for current sequence"""
        if self.instruction is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        return self.instruction

    def get_sequence_name(self):
        """Get current sequence name"""
        if self.sequence_name is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        return self.sequence_name

    def get_new_sequence_name(self):
        """Get current new sequence name (h5_file_name++sequence_name++start_end)"""
        if self.new_sequence_name is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        return self.new_sequence_name

    def get_timestamp_range(self):
        """Get the timestamp range for current sequence"""
        if self.timestamps is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        return self.timestamps[0], self.timestamps[-1]

    def transform_camera_to_2d(self, points_3d):
        """Transform 3D camera coordinates to 2D image coordinates"""
        if self.intrinsic_matrix is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        
        points_2d = self.intrinsic_matrix @ points_3d.T  # (3, 3) @ (3, n) -> (3, n)
        points_2d = points_2d[:2, :] / points_2d[2, :]  # (2, n)
        return points_2d.T

    def close(self):
        """Close H5 file"""
        if self.current_h5_data is not None:
            self.current_h5_data.close()
            self.current_h5_data = None

    def __del__(self):
        """Destructor to ensure H5 file is closed"""
        self.close()

    def list_sequences(self):
        """List all available sequences with their new sequence names"""
        for new_seq_name, (h5_path, seq_idx) in self.sequence_info.items():
            with h5py.File(h5_path, 'r') as f:
                instruction = f['instructions'][f'seq_{seq_idx}'][0].decode('utf-8')
                timestamps = f['timestamps'][f'seq_{seq_idx}'][:]
                num_frames = len(timestamps)
                start_frame, end_frame = timestamps[0], timestamps[-1]
                print(f"{new_seq_name} - {num_frames} frames ({start_frame}-{end_frame}) - {instruction}")

    def get_sequence_list(self):
        """Get list of all new sequence names"""
        return list(self.sequence_info.keys())
