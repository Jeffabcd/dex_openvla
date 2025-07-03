import os
import os.path as osp
import h5py
import glob
import numpy as np
from moviepy import VideoFileClip

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
    
    
    clip.close()

    # trasform fps to 5
    # frames = frames[::int(clip.fps/5)]
    
    return frames, clip.fps
class Hoi4d_data_func():
    def __init__(self):
       
        self.image_root = '/mnt/lab-tank/fang/hoi4d_data/HOI4D_release'
        self.hand_root = '/mnt/lab-tank/fang/hoi4d_data/Hand_pose'
        # Load all H5 files for this camera
        self.h5_files = []
        self.sequence_info = {}  # Store {new_sequence_name: (h5_file_path, seq_idx)}
        
        h5_pattern = f"/mnt/lab-home/fang/HaWoR_new/data/HOI4D/preprocess_data/HOI4D_data_*.h5"
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

    def set_sequence(self, new_sequence_name, timestamp=None):
        print('set_sequence', new_sequence_name, timestamp)
        # h5_file_name, sequence_name = new_sequence_name.split('++')
        if timestamp is not None:
            for k, v in self.sequence_info.items():
                if new_sequence_name in k:
                    start_frame = k.split('++')[-1].split('_')[0]
                    end_frame = k.split('++')[-1].split('_')[1]
                    if timestamp >= int(start_frame) and timestamp <= int(end_frame):
                        new_sequence_name = k
                        break

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
        left_hand_path = osp.join(self.hand_root,'handpose_left_hand', self.sequence_name)
        right_hand_path = osp.join(self.hand_root,'handpose_right_hand', self.sequence_name)
        is_lhand = osp.exists(left_hand_path)
        is_rhand = osp.exists(right_hand_path)
        print('='*40)
        print(f"is_lhand: {is_lhand}, is_rhand: {is_rhand}")
        if not is_lhand and not is_rhand:
            raise FileNotFoundError(f"Hand pose file not found: {left_hand_path} or {right_hand_path}")
        
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
        
        # if timestamp >= len(self.lhand_joints_cam):
        #     raise ValueError(f"Timestamp {timestamp} out of range (0-{len(self.lhand_joints_cam)-1})")
        if timestamp in self.timestamps:
            timestamp = np.where(self.timestamps == timestamp)[0][0]
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

    def get_image(self, timestamp):
        """Get original resolution image from the dataset directory"""
        if self.original_video_frames is None or self.timestamps is None:
            raise ValueError("No sequence set. Call set_sequence() first.")
        
        # if timestamp >= len(self.timestamps):
        #     raise ValueError(f"Timestamp {timestamp} out of range (0-{len(self.timestamps)-1})")
        
        # Get original timestamp and return corresponding frame
        # if timestamp in self.timestamps:
        #     timestamp = np.where(self.timestamps == timestamp)[0][0]
        # # original_timestamp = self.timestamps[timestamp]
        
        # if timestamp >= len(self.original_video_frames):
        #     raise ValueError(f"Original timestamp {timestamp} out of range for video")
        
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
