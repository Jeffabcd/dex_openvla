import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import pickle
from typing import Tuple, List, Dict, Optional
import sys
import glob
import torch

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath("/home/bidex/dex_openvla/TACO-Instructions/dataset_utils"))
from hand_pose_loader import read_hand_shape, mano_params_to_hand_info
from video_utils import mp42imgs, overlay_two_imgs
from preprocess_taco import transform_joints_world_to_camera, make_action_name

class TacoDataFunc:
    def __init__(self, dataset_root: str, object_model_root: str):
        """Initialize the TACO dataset API.
        
        Args:
            dataset_root: Root directory of the TACO dataset
            object_model_root: Root directory containing object models
        """
        self.dataset_root = dataset_root
        self.object_model_root = object_model_root
        
        # Current sequence info
        self.current_triplet = None
        self.current_sequence = None
        self.current_rgb_frames = None
        self.current_intrinsic = None
        self.current_extrinsics = None
        self.current_lhand_joints = None
        self.current_rhand_joints = None
        self.current_resolution = (1920, 1080)  # Default resolution from the dataset
        
    def set_sequence(self, triplet_sequence_name: str, timestamp: int=None):
        print('set_sequence', triplet_sequence_name, timestamp)
        """Set the current sequence to work with.
        
        Args:
            triplet: The action triplet (e.g. "(brush, brush, box)")
            sequence_name: Name of the sequence (e.g. "20231006_163")
        """
        _name = triplet_sequence_name.split('++')[-1]
        triplet, sequence_name = _name.split('/')
        self.current_triplet = triplet
        self.current_sequence = sequence_name
        
        # Load all necessary data for this sequence
        object_pose_dir = osp.join(self.dataset_root, "Object_Poses", triplet, sequence_name)
        hand_pose_dir = osp.join(self.dataset_root, "Hand_Poses", triplet, sequence_name)
        egocentric_intrinsic_filepath = osp.join(self.dataset_root, "Egocentric_Camera_Parameters", triplet, sequence_name, "egocentric_intrinsic.txt")
        egocentric_frame_extrinsic_filepath = osp.join(self.dataset_root, "Egocentric_Camera_Parameters", triplet, sequence_name, "egocentric_frame_extrinsic.npy")
        egocentric_rgb_filepath = osp.join(self.dataset_root, "Egocentric_RGB_Videos", triplet, sequence_name, "color.mp4")
        
        # Load RGB frames
        self.current_rgb_frames = mp42imgs(egocentric_rgb_filepath, return_rgb=True, max_cnt=None)
        
        # Load camera parameters
        self.current_intrinsic = np.loadtxt(egocentric_intrinsic_filepath)
        self.current_extrinsics = np.load(egocentric_frame_extrinsic_filepath)
        
        # Load hand shapes and poses
        right_hand_beta = pickle.load(open(osp.join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy()
        left_hand_beta = pickle.load(open(osp.join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        right_hand_vertices, right_hand_joints, right_hand_model_faces = mano_params_to_hand_info(osp.join(hand_pose_dir, "right_hand.pkl"), mano_beta=right_hand_beta, side="right", max_cnt=None, return_pose=False, return_faces=True, device=device)
        left_hand_vertices, left_hand_joints, left_hand_model_faces = mano_params_to_hand_info(osp.join(hand_pose_dir, "left_hand.pkl"), mano_beta=left_hand_beta, side="left", max_cnt=None, return_pose=False, return_faces=True, device=device)
      
        right_hand_joints_camera = transform_joints_world_to_camera(right_hand_joints, self.current_extrinsics)
        left_hand_joints_camera = transform_joints_world_to_camera(left_hand_joints, self.current_extrinsics)
        # Transform hand joints to camera frame
        self.current_rhand_joints = right_hand_joints_camera
        self.current_lhand_joints = left_hand_joints_camera
        
    def get_2d_joints(self, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D joint coordinates for both hands at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get joints for
            
        Returns:
            Tuple of (left_hand_joints_2d, right_hand_joints_2d), each with shape (21, 2)
        """
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        
        # Get 3D joints for the timestamp
        j_lhand = self.current_lhand_joints[timestamp_idx]
        j_rhand = self.current_rhand_joints[timestamp_idx]
        
        # Project to 2D
        current_h, current_w = self.get_image(timestamp_idx).shape[:2]
        
        # Scale intrinsic matrix according to image resize
        scale_w = current_w / self.current_resolution[0]
        scale_h = current_h / self.current_resolution[1]
        scaled_intrinsic = self.current_intrinsic.copy()
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy
        
        # Project 3D points to 2D
        j_l_2d = (scaled_intrinsic @ j_lhand.T).T  # (21, 3)
        j_l_2d = j_l_2d[:, :2] / j_l_2d[:, 2:]  # Perspective division
        j_r_2d = (scaled_intrinsic @ j_rhand.T).T
        j_r_2d = j_r_2d[:, :2] / j_r_2d[:, 2:]
        
        return j_l_2d, j_r_2d
    
    def get_3d_joints(self, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D joint coordinates for both hands at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get joints for
            
        Returns:
            Tuple of (left_hand_joints_3d, right_hand_joints_3d), each with shape (21, 3)
        """
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
            
        return self.current_lhand_joints[timestamp_idx], self.current_rhand_joints[timestamp_idx]
    
    def get_image(self, timestamp_idx: int) -> np.ndarray:
        """Get the RGB image at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get image for
            
        Returns:
            RGB image as numpy array with shape (H, W, 3)
        """
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
            
        return self.current_rgb_frames[timestamp_idx]
    
    def get_sequence_length(self) -> int:
        """Get the number of frames in the current sequence.
        
        Returns:
            Number of frames in the sequence
        """
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
            
        return len(self.current_rgb_frames)
    
    def get_instruction(self) -> str:
        """Get the instruction for the current sequence.
        
        Returns:
            Instruction string
        """
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        return make_action_name(self.current_triplet)
    
    def get_sequence_name(self) -> str:
        """Get the name of the current sequence.
        
        Returns:
            Sequence name string
        """
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        return f"{self.current_triplet}/{self.current_sequence}" 
    
    def transform_camera_to_2d(self, points: np.ndarray) -> np.ndarray:
        """Transform 3D points to 2D using camera intrinsics."""
        if self.current_sequence is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        

        
        # Project to 2D
        current_h, current_w = self.get_image(0).shape[:2]
        
        # Scale intrinsic matrix according to image resize
        scale_w = current_w / self.current_resolution[0]
        scale_h = current_h / self.current_resolution[1]
        scaled_intrinsic = self.current_intrinsic.copy()
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy
        
        # Project 3D points to 2D
        j_l_2d = (scaled_intrinsic @ points.T).T  # (21, 3)
        j_l_2d = j_l_2d[:, :2] / j_l_2d[:, 2:]  # Perspective division
        
        return j_l_2d
    
    def transform_camera_to_3d(self, points: np.ndarray) -> np.ndarray:
        """Transform 2D points to 3D using camera intrinsics."""
        return (self.current_intrinsic.inv() @ points.T).T