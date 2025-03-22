import os.path as osp
import numpy as np
import cv2
import torch
import json
import smplx
# from pytorch3d.io import load_obj, save_obj
import os

import time
import tqdm
import glob
import json
import trimesh
import pickle
from collections import Counter
import sys
sys.path.insert(0, os.path.abspath("/data2/laurence220016/Jeff/openvla/Text2HOI/"))

from lib.models.mano import build_mano_aa

from lib.utils.data import processe_params, process_egocam_result
from lib.utils.file import load_config, read_json


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
from PIL import Image

class ARCTIC_data_func():
    def __init__(self):
        arctic_config = load_config("Text2HOI/configs/dataset/arctic.yaml")
        self.data_root = arctic_config.root
        self.data_save_path = arctic_config.data_path
        self.text_root = arctic_config.text_root
        self.cropped_images_root = arctic_config.cropped_images_root
        self.images_root = arctic_config.images_root

        self.lhand_layer = build_mano_aa(is_rhand=False, flat_hand=arctic_config.flat_hand)
        self.lhand_layer = self.lhand_layer.cuda()
        self.rhand_layer = build_mano_aa(is_rhand=True, flat_hand=arctic_config.flat_hand)
        self.rhand_layer = self.rhand_layer.cuda()

    def set_sequence(self, sequence_name):
        # sequence_name: subj-obj_name
        self.sequence_name = sequence_name
        subj, obj_name = sequence_name.split("-")

        self.hand_data_path = osp.join(self.data_root, subj, f"{obj_name}.mano.npy")
        self.obj_data_path = osp.join(self.data_root, subj, f"{obj_name}.object.npy")
        self.egocam_data_path = osp.join(self.data_root, subj, f"{obj_name}.egocam.dist.npy")

        self.images_path = osp.join(self.images_root, subj, obj_name, "0")

        self.text_path = osp.join(self.text_root, subj, obj_name, "description.txt")
        
        self.lhand_joints_cam, self.rhand_joints_cam = self.get_hand_data()
        self.extrinsics, self.intrinsics = process_egocam_result(self.egocam_data_path)


    def get_hand_data(self):
        # return lhand_joints_cam, rhand_joints_cam
        hand_data, object_data = processe_params(self.hand_data_path, self.obj_data_path)
        lhand_poses = hand_data["left.pose"]        
        lhand_betas = hand_data["left.shape"] # np.zeros((lhand_poses.shape[0], 10))
        lhand_trans = hand_data["left.trans"]
        lhand_joints = process_hand_keypoints(lhand_poses, lhand_betas, lhand_trans, self.lhand_layer)
        lhand_joints = proc_numpy(lhand_joints)
        rhand_poses = hand_data["right.pose"]
        rhand_betas = hand_data["right.shape"] # np.zeros((rhand_poses.shape[0], 10))
        rhand_trans = hand_data["right.trans"]
        rhand_joints = process_hand_keypoints(rhand_poses, rhand_betas, rhand_trans, self.rhand_layer)
        rhand_joints = proc_numpy(rhand_joints)

        SE3_matrices,_ = process_egocam_result(self.egocam_data_path)
        lhand_joints_cam = transform_joints_world_to_camera(lhand_joints, SE3_matrices)
        rhand_joints_cam = transform_joints_world_to_camera(rhand_joints, SE3_matrices)
        # print('lhand_joints_cam: ', lhand_joints_cam.shape)
        return lhand_joints_cam, rhand_joints_cam
    
    def get_2d_joints(self, timestamp):
        # return lhand_joints_cam, rhand_joints_cam
        lhand_joint = self.lhand_joints_cam[timestamp]
        rhand_joint = self.rhand_joints_cam[timestamp]
        # print('lhand_joint: ', lhand_joint.shape)
        # print('intrinsics: ', self.intrinsics.shape)
        lhand_joint_2d = self.intrinsics @ lhand_joint.T #(3, 3) @ (3, 21) -> (3, 21)
        lhand_joint_2d = lhand_joint_2d[:2, :] / lhand_joint_2d[2, :] #(2, 21)
        rhand_joint_2d = self.intrinsics @ rhand_joint.T #(3, 3) @ (3, 21) -> (3, 21)
        rhand_joint_2d = rhand_joint_2d[:2, :] / rhand_joint_2d[2, :] #(2, 21)
        return lhand_joint_2d.T, rhand_joint_2d.T   
    def get_3d_joints(self, timestamp):
        lhand_joint = self.lhand_joints_cam[timestamp]
        rhand_joint = self.rhand_joints_cam[timestamp]
        return lhand_joint, rhand_joint
    
    def transform_camera_to_2d(self, points_3d):
        points_2d = self.intrinsics @ points_3d.T #(3, 3) @ (3, n) -> (3, n)
        points_2d = points_2d[:2, :] / points_2d[2, :] #(2, n)
        return points_2d.T
    
    def get_image(self, timestamp):
        img_path = osp.join(self.images_path, f"{timestamp+1:05d}.jpg")
        img = np.array(Image.open(img_path))
        return img
    
