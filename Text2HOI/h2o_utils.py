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
import sys
#from moviepy.editor import ImageSequenceClip
sys.path.insert(0, os.path.abspath("/home/bidex/dex_openvla/Text2HOI/"))

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


class h2o_data_func():
    def __init__(self):

        h2o_config = load_config("Text2HOI/configs/dataset/h2o.yaml")
        self.data_root = h2o_config.root
        self.data_save_path = h2o_config.data_path

        lhand_layer = build_mano_aa(is_rhand=False, flat_hand=h2o_config.flat_hand)
        self.lhand_layer = lhand_layer.cuda()
        rhand_layer = build_mano_aa(is_rhand=True, flat_hand=h2o_config.flat_hand)
        self.rhand_layer = rhand_layer.cuda()

    def set_sequence(self, sequence_name, timestamp=None):
        # sequence_name : data_path
        if not osp.exists(sequence_name):
            s_list = sequence_name.split('tank')
            if len(s_list) > 2:
                raise
            sequence_name = 'lab-tank'.join(s_list)
            
        hand_pose_manos, obj_pose_rts, cam_poses, action_labels, images, intrinsic_matrix, resolution \
            = get_data_path_h2o(sequence_name)

        self.hand_pose_paths = hand_pose_manos
        self.obj_pose_paths = obj_pose_rts
        self.cam_pose_paths = cam_poses
        self.action_labels = action_labels
        self.images = images
        self.intrinsic_matrix = intrinsic_matrix
        self.resolution = resolution

        self.sequence_name = sequence_name

    
    def get_2d_joints(self, timestamp):
        # return lhand_joints_cam, rhand_joints_cam
        j_lhand, j_rhand = self.get_3d_joints(timestamp)
        # (21, 3)

        current_h, current_w = self.get_image(timestamp).shape[:2]
    
        # Scale intrinsic matrix according to image resize
        scale_w = current_w / self.resolution[0]
        scale_h = current_h / self.resolution[1]
        scaled_intrinsic = self.intrinsic_matrix.copy()
        #print('scale', scale_w, scale_h)
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy

        # print('lhand_joint: ', lhand_joint.shape)
        # print('intrinsics: ', self.intrinsics.shape)
        j_l_2d = (scaled_intrinsic @ j_lhand.T).T  # (21, 3)
        j_l_2d = j_l_2d[:, :2] / j_l_2d[:, 2:]  # Perspective division
        j_r_2d = (scaled_intrinsic @ j_rhand.T).T
        j_r_2d = j_r_2d[:, :2] / j_r_2d[:, 2:]
        return j_l_2d, j_r_2d   

    def get_3d_joints(self, timestamp):
        hand_pose_mano_data = np.loadtxt(self.hand_pose_paths[timestamp])
        obj_pose_rt_data = np.loadtxt(self.obj_pose_paths[timestamp])
        extrinsic_matrix = np.loadtxt(self.cam_pose_paths[timestamp]).reshape(4, 4)

        # left hand
        lhand_trans = hand_pose_mano_data[1:4]
        lhand_pose = hand_pose_mano_data[4:52]
        lhand_beta = hand_pose_mano_data[52:62]
        # transform hand pose to camera coordinate
        left_rotvec = process_hand_pose_h2o(lhand_pose, lhand_trans, extrinsic_matrix)
        lhand_pose[:3] = left_rotvec
        new_left_trans, lhand_origin = process_hand_trans_h2o(lhand_pose, lhand_beta, lhand_trans, extrinsic_matrix, self.lhand_layer)
  
        # right hand
        rhand_trans = hand_pose_mano_data[63:66]
        rhand_pose = hand_pose_mano_data[66:114]
        rhand_beta = hand_pose_mano_data[114:124]
        # transform hand pose to camera coordinate
        right_rotvec = process_hand_pose_h2o(rhand_pose, rhand_trans, extrinsic_matrix)
        rhand_pose[:3] = right_rotvec
        new_right_trans, rhand_origin = process_hand_trans_h2o(rhand_pose, rhand_beta, rhand_trans, extrinsic_matrix, self.rhand_layer)

        x_lhand = transform_hand_to_xdata([new_left_trans], [lhand_pose])
        x_rhand = transform_hand_to_xdata([new_right_trans], [rhand_pose])
        j_lhand = transform_xdata_to_joints(x_lhand, self.lhand_layer)
        j_rhand = transform_xdata_to_joints(x_rhand, self.rhand_layer)

        return j_lhand[0], j_rhand[0]
    
    def transform_camera_to_2d(self, points_3d):
        current_h, current_w = self.get_image(0).shape[:2]
    
        # Scale intrinsic matrix according to image resize
        scale_w = current_w / self.resolution[0]
        scale_h = current_h / self.resolution[1]
        scaled_intrinsic = self.intrinsic_matrix.copy()
        #print('scale', scale_w, scale_h)
        # scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        # scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy
        points_2d = scaled_intrinsic @ points_3d.T #(3, 3) @ (3, n) -> (3, n)

        points_2d = points_2d[:2, :] / points_2d[2, :] #(2, n)
        return points_2d.T
    
    def get_image(self, timestamp):

        img_path = self.images[timestamp]
        img = np.array(Image.open(img_path))
        return img
    
