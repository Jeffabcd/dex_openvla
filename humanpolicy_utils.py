import os
import os.path as osp
import h5py
import glob
import cv2
import numpy as np
from PIL import Image

class HumanpolicyDataFunc():
    def __init__(self):
        # Initialize paths
        self.image_root = '/mnt/lab-home/fang/human_policy/downloads'
        
        # Load all H5 files
        self.h5_files = []
        self.sequence_info = {}  # Store {new_sequence_name: (h5_file_path, seq_idx)}
        
        h5_pattern = f"data/humanpolicy/preprocess_data/humanpolicy_data_*.h5"
        h5_paths = glob.glob(h5_pattern)
        h5_paths.sort()
        
        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as f:
                num_sequences = f.attrs['num_sequences']
                for seq_idx in range(num_sequences):
                    sequence_name = f['sequence_names'][f'seq_{seq_idx}'][0].decode('utf-8')
                    timestamps = f['timestamps'][f'seq_{seq_idx}'][:]
                    start_frame = timestamps[0]
                    end_frame = timestamps[-1]
                    # Create unique identifier with timestamp range
                    file_name = osp.basename(h5_path).split('.')[0]
                    print('lksjlkjlskdjflk')
                    print(file_name)
                    new_sequence_name = f"{file_name}++{sequence_name}++{start_frame}_{end_frame}"
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

    def set_sequence(self, new_sequence_name, timestamp=None):
        """Set the current sequence by new_sequence_name (h5_file_name++sequence_name++start_end)"""
        print('set_sequence', new_sequence_name, timestamp)
        h5_file_name = new_sequence_name.split('++')[0]
        sequence_name = new_sequence_name.split('++')[1]
        task_id, proceed_id = sequence_name.split('/')
        proceed_id = proceed_id.split('.')[0]
        
            
        # h5_path = self.sequence_info[h5_file_name][0]
        # seq_idx = self.sequence_info[h5_file_name][1]
        new_sequence_name = h5_file_name + '++' + sequence_name
        
        if timestamp is not None:
            for k, v in self.sequence_info.items():
                if new_sequence_name in k:
                    start_frame = k.split('++')[-1].split('_')[0]
                    end_frame = k.split('++')[-1].split('_')[1]
                    start_frame = int(start_frame)
                    end_frame = int(end_frame)
                    timestamp = int(timestamp)
                    if timestamp >= start_frame and timestamp <= end_frame:
                        new_sequence_name = k
                        break

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
        video_path = osp.join(self.image_root, task_id, 'video', f'{proceed_id}.mp4')
        
        if not osp.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Only load video if it's different from current one
        if self.original_video_frames is None or self.current_video_path != video_path:
            self.original_video_frames = self._load_video_frames(video_path)
            self.current_video_path = video_path
        
        print(f"Set sequence: {new_sequence_name}")
        print(f"Original sequence: {self.sequence_name}")
        print(f"Frames: {len(self.timestamps)} (from {self.timestamps[0]} to {self.timestamps[-1]})")
        print(f"Instruction: {self.instruction}")

    def _load_video_frames(self, video_path):
        """Load frames from original video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        # switch fps from 30 to 5
        frames = frames[::6]
        return frames

    def get_3d_joints(self, timestamp):
        """Get 3D joints for left and right hands at given timestamp"""
        if self.lhand_joints_cam is None or self.rhand_joints_cam is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        
    

        if timestamp in self.timestamps:
            timestamp = np.where(self.timestamps == timestamp)[0][0]

        if timestamp >= len(self.lhand_joints_cam):
            raise ValueError(f"Timestamp {timestamp} out of range (0-{len(self.lhand_joints_cam)-1})")
        j_lhand = self.lhand_joints_cam[timestamp]  # (21, 3)
        j_rhand = self.rhand_joints_cam[timestamp]  # (21, 3)
        
        return j_lhand, j_rhand

    def get_2d_joints(self, timestamp):
        """Get 2D joints for left and right hands at given timestamp"""
        j_lhand, j_rhand = self.get_3d_joints(timestamp)
        
        # Get current image dimensions from original video
        if timestamp in self.timestamps:
            timestamp = np.where(self.timestamps == timestamp)[0][0]

        current_img = self.original_video_frames[timestamp]
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
    def transform_camera_to_2d(self, points_3d):
        """Transform 3D points to 2D points"""
        current_img = self.original_video_frames[0]
        current_h, current_w = current_img.shape[:2]
        
        # Scale intrinsic matrix according to image resize
        scale_w = current_w / self.resolution[0]
        scale_h = current_h / self.resolution[1]
        scaled_intrinsic = self.intrinsic_matrix.copy()
        scaled_intrinsic[0, :] *= scale_w  # Scale fx and cx
        scaled_intrinsic[1, :] *= scale_h  # Scale fy and cy
        
        # Project 3D points to 2D
        j_l_2d = (scaled_intrinsic @ points_3d.T).T  # (21, 3)
        j_l_2d = j_l_2d[:, :2] / j_l_2d[:, 2:]  # Perspective division
        return j_l_2d

    def get_image(self, timestamp):
        """Get original resolution image from the dataset directory"""
        if self.original_video_frames is None or self.timestamps is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
       
        
        if timestamp in self.timestamps:
            timestamp = np.where(self.timestamps == timestamp)[0][0]
        
        # Get original timestamp and return corresponding frame
            
        
        if timestamp >= len(self.original_video_frames):
            raise ValueError(f"Original timestamp {timestamp} out of range for video")
        
        return self.original_video_frames[timestamp]

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