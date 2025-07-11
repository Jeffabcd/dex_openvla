import os
import sys
import numpy as np
import cv2
import pickle
import torch
from typing import Tuple, List, Dict, Optional
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath("/home/bidex/dex_openvla/OakInk2/"))
from manotorch.manolayer import ManoLayer
from src.oakink2_toolkit.dataset import OakInk2__Dataset
from src.oakink2_toolkit.structure import OakInk2__PrimitiveTask
from src.oakink2_toolkit.meta import VIDEO_SHAPE
from src.oakink2_toolkit.tool import slice_param

class OakInk2DataFunc:
    def __init__(self, dataset_root: str, mano_path: str):
        """Initialize the OakInk2 dataset API.
        
        Args:
            dataset_root: Root directory of the OakInk2 dataset
            mano_path: Path to MANO model files
        """
        self.dataset_root = dataset_root
        self.mano_path = mano_path
        self.CAM_NAME = "egocentric"
        
        # Initialize MANO layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
        # Initialize right hand MANO model
        self.mano_layer_rh = ManoLayer(
            mano_assets_root=mano_path,
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        ).to(self.device)
        self.hand_faces_rh = self.mano_layer_rh.get_mano_closed_faces().cpu().numpy()
        
        # Initialize left hand MANO model
        self.mano_layer_lh = ManoLayer(
            mano_assets_root=mano_path,
            rot_mode="quat",
            side="left",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        ).to(self.device)
        self.hand_faces_lh = self._face_lh(self.mano_layer_lh)
        


        # print('dataset_root', dataset_root)
        # Initialize dataset
        self.oakink2_dataset = OakInk2__Dataset(
            dataset_prefix=dataset_root,
            return_instantiated=True,
        )
        
        # Current sequence info
        self.current_seq_key = None
        self.current_primitive_task = None
        self.current_cam_intr = None
        self.current_cam_extr = None
        self.current_frame_id_list = None
        self.current_cam_def = None
        self.current_cam_revdef = None
        self.current_lhand_joints = None
        self.current_rhand_joints = None
        self.current_images = None
        
    def _face_lh(self, mano_layer_lh):
        """Get faces for left hand MANO model."""
        _close_faces = torch.Tensor([
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ])
        _th_closed_faces = torch.cat([mano_layer_lh.th_faces.clone().detach().cpu(), 
                                    _close_faces[:, [2, 1, 0]].long()])
        return _th_closed_faces.cpu().numpy()
        
    def set_sequence(self, seq_key: str, timestamp: int=None):
        """Set the current sequence to work with.
        
        Args:
            seq_key: The sequence key (e.g. "seq1/seq2")
            primitive_identifier: The primitive task identifier
        """
        primitive_identifier = None
        # print('set_sequence', seq_key, primitive_identifier)
        primitive_id  = int(seq_key.split('_')[-1])
        file, seq_key = seq_key.split('++')
        last_part_id = len(seq_key)-1
        while seq_key[last_part_id] != '_':
            last_part_id -= 1
        seq_key = seq_key[:last_part_id]
        self.current_seq_key = seq_key
        # if timestamp is not None:
        #     primitive_identifier = f"{primitive_identifier}_{timestamp}"
        
        # Load complex task data
        complex_task_data = self.oakink2_dataset.load_complex_task(seq_key)
        
        # Load primitive task data
        primitive_task_data_list = self.oakink2_dataset.load_primitive_task(
            complex_task_data, primitive_identifier)
        if isinstance(primitive_task_data_list, OakInk2__PrimitiveTask):
            primitive_task_data_list = [primitive_task_data_list]
        
        # Load camera parameters
        anno_filepath = os.path.join(self.oakink2_dataset.anno_prefix, 
                                   f"{complex_task_data.seq_token}.pkl")
        with open(anno_filepath, "rb") as ifs:
            anno_info = pickle.load(ifs)
            
        self.current_cam_intr = next(iter(anno_info["cam_intr"][self.CAM_NAME].values()))
        self.current_cam_extr = anno_info["cam_extr"][self.CAM_NAME]
        self.current_frame_id_list = anno_info["frame_id_list"]
        self.current_cam_def = anno_info["cam_def"]
        self.current_cam_revdef = {v: k for k, v in self.current_cam_def.items()}
        
        
        # Process each primitive task
        ptask_data = primitive_task_data_list[primitive_id]
        self.current_primitive_task = ptask_data
        
        # Process left hand
        lh_out = self._process_hand_data(ptask_data, "left")
        
        # Process right hand
        rh_out = self._process_hand_data(ptask_data, "right")
        
        # Load images and process joints
        self._process_sequence_data(ptask_data, lh_out, rh_out)
            
    def _process_hand_data(self, ptask_data, side: str) -> Dict:
        """Process hand data for given side."""
        out = {}
        if side == "left" and ptask_data.frame_range_lh is not None:
            j_list, v_list = [], []
            lh_pose_info = {}
            # index pose_info with mask
            for k, v in ptask_data.lh_param.items():
                lh_pose_info[k] = v[ptask_data.lh_in_range_mask].to(dtype=self.dtype, device=self.device)
            for lh_pose_item in slice_param(lh_pose_info, batch_size=100):
                mano_out_sl = self.mano_layer_lh(pose_coeffs=lh_pose_item["pose_coeffs"], betas=lh_pose_item["betas"])
                j_sl = mano_out_sl.joints + lh_pose_item["tsl"].unsqueeze(1)
                v_sl = mano_out_sl.verts + lh_pose_item["tsl"].unsqueeze(1)
                j = j_sl.clone().cpu().numpy()
                v = v_sl.clone().cpu().numpy()

                j_list.append(j)
                v_list.append(v)
            j_traj = np.concatenate(j_list, axis=0)
            v_traj = np.concatenate(v_list, axis=0)
            # print('j_traj ', j_traj.shape)
            out["j"] = j_traj
            out["v"] = v_traj

            
        elif side == "right" and ptask_data.frame_range_rh is not None:
            pose_info = {}
            for k, v in ptask_data.rh_param.items():
                pose_info[k] = v[ptask_data.rh_in_range_mask].to(dtype=self.dtype, device=self.device)
            
            j_list, v_list = [], []
            mano_layer = self.mano_layer_rh
            
            for pose_item in slice_param(pose_info, batch_size=100):
                mano_out = mano_layer(pose_coeffs=pose_item["pose_coeffs"], 
                                    betas=pose_item["betas"])
                j = mano_out.joints + pose_item["tsl"].unsqueeze(1)
                v = mano_out.verts + pose_item["tsl"].unsqueeze(1)
                j_list.append(j.clone().cpu().numpy())
                v_list.append(v.clone().cpu().numpy())
                
            out["j"] = np.concatenate(j_list, axis=0)
            out["v"] = np.concatenate(v_list, axis=0)
            
        return out
    
    def _process_sequence_data(self, ptask_data, lh_out: Dict, rh_out: Dict):
        """Process sequence data including images and joints."""
        viz_step = 15
        fid_list = [
            self.current_frame_id_list[np.argmin(np.abs(np.array(self.current_frame_id_list) - _f))]
            for _f in range(ptask_data.frame_range[0], ptask_data.frame_range[1], viz_step)
        ]
        self.fid_list = fid_list
        # print('fid_list', fid_list)
        
        j_lhand_list, j_rhand_list = [], []
        j_world_lhand, j_world_rhand = [], []
        img_list = []
        
        for fid in fid_list:
            cam_extr = self.current_cam_extr[fid]
            offset = fid - ptask_data.frame_range[0]
            
            # Process left hand joints
            if ptask_data.frame_range_lh is not None:
                offset_lh = fid - ptask_data.frame_range_lh[0]
                if 0 <= offset_lh and offset_lh < lh_out["j"].shape[0]:
                    j_lh = lh_out["j"][offset_lh]
                    j_world_lhand.append(j_lh)
                    j_lh = self._transform_points(j_lh, cam_extr)
                    j_lhand_list.append(j_lh)
                    
            # Process right hand joints
            if ptask_data.frame_range_rh is not None:
                offset_rh = fid - ptask_data.frame_range_rh[0]
                if 0 <= offset_rh and offset_rh < rh_out["j"].shape[0]:
                    j_rh = rh_out["j"][offset_rh]
                    j_world_rhand.append(j_rh)
                    j_rh = self._transform_points(j_rh, cam_extr)
                    j_rhand_list.append(j_rh)
            
            # Load image
            img = cv2.imread(os.path.join(
                self.oakink2_dataset.data_prefix,
                ptask_data.seq_token,
                self.current_cam_revdef[self.CAM_NAME],
                f"{fid:0>6}.png"
            ))
            img_list.append(img)
        
        # Convert lists to arrays and handle missing hands
        img_list = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list])
        # img_list = np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4) 
        #                     for img in img_list])
        
        if len(j_lhand_list) == 0:
            j_lhand_list = np.zeros((len(j_rhand_list), 21, 3)) 
            j_world_lhand = np.zeros((len(j_rhand_list), 21, 3)) 
        else:
            j_lhand_list = np.array(j_lhand_list)
            j_world_lhand = np.array(j_world_lhand)
            
        if len(j_rhand_list) == 0:
            j_rhand_list = np.zeros((len(j_lhand_list), 21, 3))
            j_world_rhand = np.zeros((len(j_lhand_list), 21, 3))
        else:
            j_rhand_list = np.array(j_rhand_list)
            j_world_rhand = np.array(j_world_rhand)
        
        # Store processed data
        self.current_images = img_list
        self.current_lhand_joints = j_lhand_list
        self.current_rhand_joints = j_rhand_list
        self.world_lhand_joints = j_world_lhand
        self.world_rhand_joints = j_world_rhand
    
    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform points using transformation matrix."""

        leading_shape = points.shape[:-2]
        leading_dim = len(leading_shape)

        res = (
            np.swapaxes(
                np.matmul(
                    transform[..., :3, :3],
                    np.swapaxes(points, leading_dim, leading_dim + 1),
                ),
                leading_dim,
                leading_dim + 1,
            )  # [..., X, 3]
            + transform[..., :3, 3][..., np.newaxis, :]  # [..., 1, 3]
        )
        return res
        #return (transform @ np.vstack((points.T, np.ones(points.shape[0]))))[:3].T
    
    def get_2d_joints(self, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D joint coordinates for both hands at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get joints for
            
        Returns:
            Tuple of (left_hand_joints_2d, right_hand_joints_2d), each with shape (21, 2)
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        
        # Get 3D joints
        j_lhand, j_rhand = self.get_3d_joints(timestamp_idx)
        h, w  = self.get_image(0).shape[:2]
        self.resolution = (w, h)
        # Project to 2D
        current_h, current_w = self.get_image(timestamp_idx).shape[:2]
        scale_w = current_w / self.resolution[0]
        scale_h = current_h / self.resolution[1]
        # print('get_2d_joints', scale_w, scale_h)
        scaled_intrinsic = self.current_cam_intr.copy()
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy

        # Project 3D points to 2D
        j_l_2d = (scaled_intrinsic @ j_lhand.T).T  # (21, 3)
        z_l = j_l_2d[:, 2:]
        z_l[np.abs(z_l) < 1e-7] = 1e-7
        j_l_2d = j_l_2d[:, :2] / z_l  # Perspective division
        j_r_2d = (scaled_intrinsic @ j_rhand.T).T
        z_r = j_r_2d[:, 2:]
        z_r[np.abs(z_r) < 1e-7] = 1e-7
        j_r_2d = j_r_2d[:, :2] / z_r  # Perspective division
        
        return j_l_2d, j_r_2d
    
    def get_3d_joints(self, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D joint coordinates for both hands at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get joints for
            
        Returns:
            Tuple of (left_hand_joints_3d, right_hand_joints_3d), each with shape (21, 3)
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        #index = self.current_frame_id_list.index(timestamp_idx)
        if timestamp_idx in self.fid_list:
            index = self.fid_list.index(timestamp_idx)
        else:
            index = timestamp_idx
        return self.current_lhand_joints[index], self.current_rhand_joints[index]
    
    def get_world_3d_joints(self, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D joint coordinates for both hands at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get joints for
            
        Returns:
            Tuple of (left_hand_joints_3d, right_hand_joints_3d), each with shape (21, 3)
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        #index = self.current_frame_id_list.index(timestamp_idx)
        if timestamp_idx in self.fid_list:
            index = self.fid_list.index(timestamp_idx)
        else:
            index = timestamp_idx
        return self.world_lhand_joints[index], self.world_rhand_joints[index]

    def get_action(self, timestamp_idx: int, next_timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
       
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        #index = self.current_frame_id_list.index(timestamp_idx)
        if timestamp_idx in self.fid_list:
            index = self.fid_list.index(timestamp_idx)
            n_index = self.fid_list.index(next_timestamp_idx)
            # print(f'n index: {n_index}, index: {index}')
        else:
            index = timestamp_idx


        is_lhand = np.any(self.world_lhand_joints)
        is_rhand = np.any(self.world_rhand_joints)
        # print('dataset state: ', self.current_lhand_joints[index][0], ' ', self.current_rhand_joints[index][0])
        # print(f'is lhand: {is_lhand}, is rhand: {is_rhand}')
        c_extr = self.current_cam_extr[timestamp_idx]
        if is_lhand:
            # transform to camera frame
            c_lhand_joints = self._transform_points(self.world_lhand_joints[index], c_extr)
            # the position of t=i & t=i+1 should share same cam_extr at t=i
            n_lhand_joints = self._transform_points(self.world_lhand_joints[n_index], c_extr)
            lhand_actions = n_lhand_joints - c_lhand_joints
            
        else:
            lhand_actions = self.world_lhand_joints[index]

        if is_rhand:
            # transform to camera frame
            c_rhand_joints = self._transform_points(self.world_rhand_joints[index], c_extr)
            # the position of t=i & t=i+1 should share same cam_extr at t=i
            n_rhand_joints = self._transform_points(self.world_rhand_joints[n_index], c_extr)
            rhand_actions = n_rhand_joints - c_rhand_joints
            
        else:
            rhand_actions = self.world_rhand_joints[index]

        # print('l hand action: ', lhand_actions, ' r hand action: ', rhand_actions)
        
        return [torch.cat([torch.tensor(lhand_actions), torch.tensor(rhand_actions)], dim=-1)]

    

        
    def get_extrinsic(self, timestamp_idx: int) -> np.ndarray:
        """Get the extrinsic matrix at the specified timestamp.
        Args:
            timestamp_idx: Index of the timestamp to get extrinsic for
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")

        index = timestamp_idx
        return self.current_cam_extr[index]

    
    def get_image(self, timestamp_idx: int) -> np.ndarray:
        """Get the RGB image at the specified timestamp.
        
        Args:
            timestamp_idx: Index of the timestamp to get image for
            
        Returns:
            RGB image as numpy array with shape (H, W, 3)
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        if timestamp_idx in self.fid_list:
            index = self.fid_list.index(timestamp_idx)
        else:
            index = timestamp_idx
        return self.current_images[index]
    
    def get_sequence_length(self) -> int:
        """Get the number of frames in the current sequence.
        
        Returns:
            Number of frames in the sequence
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        
        return len(self.current_images)
    
    def get_instruction(self) -> str:
        """Get the instruction for the current sequence.
        
        Returns:
            Instruction string
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        return self.current_primitive_task.task_desc
    
    def get_sequence_name(self) -> str:
        """Get the name of the current sequence.
        
        Returns:
            Sequence name string
        """
        if self.current_primitive_task is None:
            raise ValueError("No sequence selected. Call set_sequence first.")
        return self.current_seq_key 
    
    def transform_camera_to_2d(self, points: np.ndarray) -> np.ndarray:
        """Transform 3D points to 2D using camera intrinsics."""
        current_h, current_w = self.get_image(0).shape[:2]
        h, w  = self.get_image(0).shape[:2]
        self.resolution = (w, h)
        scale_w = current_w / self.resolution[0]
        scale_h = current_h / self.resolution[1]
        # print('transform_camera_to_2d', scale_w, scale_h)
        scaled_intrinsic = self.current_cam_intr.copy()
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy

        # Project 3D points to 2D
        j_2d = (scaled_intrinsic @ points.T).T  # (21, 3)
        z = j_2d[:, 2:]
        z[np.abs(z) < 1e-7] = 1e-7
        j_2d = j_2d[:, :2] / z  # Perspective division
        return j_2d
    
    def transform_camera_to_3d(self, points: np.ndarray) -> np.ndarray:
        """Transform 2D points to 3D using camera intrinsics."""
        return (self.current_cam_revdef @ points.T).T