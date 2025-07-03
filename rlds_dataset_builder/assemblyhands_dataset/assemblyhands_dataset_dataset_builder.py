from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import cv2
from PIL import Image

class AssemblyHandsDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('8'*50)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'hand_state': tfds.features.Tensor(
                            shape=(126,),
                            dtype=np.float32,
                            doc='Robot state, consists of [x_L,y_L,z_L, x_R, y_R, z_R]',
                        ),
                        'wrist_state': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot state, consists of [x_L,y_L,z_L, x_R, y_R, z_R]',
                        ),
                       # 'wrist_image': tfds.features.Image(
                       #     shape=(64, 64, 3),
                       #     dtype=np.uint8,
                       #     encoding_format='png',
                       #     doc='Wrist camera RGB observation.',
                       # ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot action, consists of [dx_L,dy_L,dz_L, dx_R, dy_R, dz_R]',
                    ),
                    'action_spherical': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot action in spherical coordinates, consists of [theta, phi, r]',
                    ),
                    'wrist_action': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot action, consists of [dx_L,dy_L,dz_L, dx_R, dy_R, dz_R]',
                    ),
                    'hand_action': tfds.features.Tensor(
                        shape=(126,),
                        dtype=np.float32,
                        doc='Robot action, consists of [dx_L,dy_L,dz_L, dx_R, dy_R, dz_R]',
                    ),
                    'intrinsics': tfds.features.Tensor(
                        shape=(3, 3),
                        dtype=np.float32,
                        doc='Intrinsics matrix',
                    ),
                    'ori_width_height': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.int32,
                        doc='Original width and height of the image',
                    ),
                    'sequence_name': tfds.features.Text(
                        doc='sequence_name'
                    ),
                    'timestamp_ns': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='timestamp_ns',
                    ),
                    'next_timestamp_ns': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='next timestamp_ns',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
       # dataset_name = 'world_frame_50'
        self.perm = None
        return {
            'train': self._generate_examples(path=f'/mnt/lab-home/fang/assemblyhands-toolkit/preprocess_data', split='train'),
            'val': self._generate_examples(path=f'/mnt/lab-home/fang/assemblyhands-toolkit/preprocess_data', split='val'),
            #'val': self._generate_examples(path='data/val/episode_*.npy'),
        }
    
    def cartesian_to_spherical(self, translation):
        # translation: (6,)
        print('=========================')
        print(translation)
        print('========99999=================')
        translations = [translation[:3], translation[3:]]
        spherical_actions = []
        for translation in translations:
            x, y, z = translation[0], translation[1], translation[2]
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # polar angle
            phi = np.arctan2(y, x)  # azimuthal angle
            r = np.sqrt(x**2 + y**2 + z**2)
            spherical_actions.append(np.array([theta, phi, r]).astype(np.float32))
            
        result = np.concatenate(spherical_actions, axis=0)
        print(result)
        return result

    def _generate_examples(self, path, split='') -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        def _parse_example(sample_idx, path, seq_idx):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            with h5py.File(path, 'r') as f:
                images = f['images'][f'seq_{seq_idx}'][:]
        
                # Get instruction (decode from bytes to string)
                instruction = f['instructions'][f'seq_{seq_idx}'][:][0].decode('utf-8')
                instruction = 'Do something'
                
                # Get hand joints
                lhand_joints = f['lhand_joints'][f'seq_{seq_idx}'][:]
                rhand_joints = f['rhand_joints'][f'seq_{seq_idx}'][:]
                
                # Get sequence name
                sequence_name = f['sequence_names'][f'seq_{seq_idx}'][:][0].decode('utf-8')
                
                # Get timestamps
                timestamps = f['timestamps'][f'seq_{seq_idx}'][:]
                intrinsics = f['intrinsics'][f'seq_{seq_idx}'][:]
                #print(intrinsics.shape)
                resolution = f['resolutions'][f'seq_{seq_idx}'][:]
                # ori_width = f.attrs['ori_width']
                # ori_height = f.attrs['ori_height']
            joint_states = np.concatenate([lhand_joints, rhand_joints], axis=1) # (N, 42, 3)
            joint_states = joint_states.reshape(joint_states.shape[0], -1) # (N, 126)
            joint_actions = joint_states[1:] - joint_states[:-1] # (N-1, 126)
            joint_actions = np.concatenate([joint_actions, np.zeros((1, joint_states.shape[1]))], axis=0) # (N, 126)
             
            wrist_states = np.concatenate([lhand_joints[:, 0], rhand_joints[:, 0]], axis=1) # (N, 6)
            wrist_actions = wrist_states[1:] - wrist_states[:-1] # (N-1, 6)
            wrist_actions = np.concatenate([wrist_actions, np.zeros((1, wrist_states.shape[1]))], axis=0) # (N, 6)


            if wrist_states.shape[0] != wrist_actions.shape[0]:
                print(f'{path} dimension mismatch!')
                raise

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, _image in enumerate(images):
                # compute Kona language embedding
                language_embedding = self._embed([instruction])[0].numpy()
                # convert PIL image to cv2 image
                #img_resized_cv2 = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR) 
                #rotated_image = cv2.rotate(img_resized_cv2, cv2.ROTATE_90_CLOCKWISE)
                
                #pre_image = cv2.resize(rotated_image, (256, 256), interpolation=cv2.INTER_AREA)
                #rgb_image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                if _image.shape[0] > 256:
                    img = Image.fromarray(_image)
                    rgb_image = np.array(img.resize((256, 256), Image.LANCZOS))
                else:
                    rgb_image = _image
                
                episode.append({
                    'observation': {
                        'image': rgb_image,
                        'wrist_state': wrist_states[i].astype(np.float32),
                        'hand_state': joint_states[i].astype(np.float32),
                    },
                    'wrist_action': wrist_actions[i].astype(np.float32), 
                    'hand_action': joint_actions[i].astype(np.float32), 
                    'action': wrist_actions[i].astype(np.float32),
                    'action_spherical': self.cartesian_to_spherical(wrist_actions[i]).astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(wrist_states) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(wrist_states) - 1),
                    'is_terminal': i == (len(wrist_states) - 1),
                    'language_instruction': instruction,
                    'sequence_name': sequence_name,
                    'timestamp_ns': timestamps[i],
                    'next_timestamp_ns': timestamps[i+1] if i < (len(wrist_states)-1) else timestamps[i],
                    'language_embedding': language_embedding,
                    'intrinsics': intrinsics.astype(np.float32),
                    'ori_width_height': np.array([resolution[0], resolution[1]], dtype=np.int32)
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return sample_idx, sample

        # create list of all examples
        h5_files = glob.glob(f'{path}/*.h5')
        h5_files = sorted(h5_files)
        print('='*30)
        print('h5_files: ', h5_files)
        print('='*30)
        file_index = []
        total_length = 0
        new_h5_files = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                num_sequences = f.attrs['num_sequences']
                if num_sequences == 0:
                    continue
                fps = f.attrs['fps']
                original_fps = f.attrs['original_fps']
                print(f"Dataset contains {num_sequences} sequences")
                print(f"FPS: {fps} (original: {original_fps})")

                file_index.append(total_length)
                total_length += num_sequences
                new_h5_files.append(h5_file)

        print('total length: ', total_length)
        print('file index: ', file_index)

        if self.perm is None:
            np.random.seed(42)
            self.perm = np.random.permutation(total_length)

        #current_sequence_name = ['s01-box_grab_01', 's01-box_use_01', 's01-box_use_02', 's01-capsulemachine_grab_01', 's01-capsulemachine_use_01']

        # for smallish datasets, use single-thread parsing
        # for sample in list(perm[:int(0.9*num_sequences)]):
        #     with h5py.File(path, 'r') as f:
        #         sequence_name = f['sequence_names'][f'seq_{sample}'][:][0].decode('utf-8')
        #     if sequence_name in current_sequence_name:
        #         yield _parse_example(int(sample), path)


        if split == 'train':
            start_idx = 0
            end_idx = int(0.9*total_length)
        else:
            start_idx = int(0.9*total_length)
            end_idx = total_length

        for sample in list(self.perm[start_idx:end_idx]):
            file_idx = np.searchsorted(file_index, sample) if sample in file_index else np.searchsorted(file_index, sample) - 1
            print('sample: ', sample, 'file_idx: ', file_idx)

            h5_file = new_h5_files[file_idx]
            sample_idx = sample - file_index[file_idx]
            yield _parse_example(int(sample), h5_file, int(sample_idx))
        
        # for sample in list(range(num_sequences)):
        #     yield _parse_example(int(sample), path)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

