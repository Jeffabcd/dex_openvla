from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import cv2

class ArcticDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        dataset_name = 'camera_frame_300_fps5'
        return {
            'train': self._generate_examples(path=f'../../arctic/preprocess_data/arctic_data_world.h5', split='train'),
            #'val': self._generate_examples(path='../../hot3d/hot3d/preprocess_dataset/{dataset_name}/*/*.hdf5', split='val'),
            #'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path, split='') -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        def _parse_example(seq_idx, path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            with h5py.File(path, 'r') as f:
                images = f['images'][f'seq_{seq_idx}'][:]
        
                # Get instruction (decode from bytes to string)
                instruction = f['instructions'][f'seq_{seq_idx}'][:][0].decode('utf-8')
                
                # Get hand joints
                lhand_joints = f['lhand_joints'][f'seq_{seq_idx}'][:]
                rhand_joints = f['rhand_joints'][f'seq_{seq_idx}'][:]
                
                # Get sequence name
                sequence_name = f['sequence_names'][f'seq_{seq_idx}'][:][0].decode('utf-8')
                
                # Get timestamps
                timestamps = f['timestamps'][f'seq_{seq_idx}'][:]
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
                rgb_image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                
                episode.append({
                    'observation': {
                        'image': rgb_image,
                        'wrist_state': wrist_states[i].astype(np.float32),
                        'hand_state': joint_states[i].astype(np.float32),
                    },
                    'wrist_action': wrist_actions[i].astype(np.float32), 
                    'hand_action': joint_actions[i].astype(np.float32), 
                    'action': wrist_actions[i].astype(np.float32),
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
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return seq_idx, sample

        # create list of all examples
        with h5py.File(path, 'r') as f:
            num_sequences = f.attrs['num_sequences']
            fps = f.attrs['fps']
            original_fps = f.attrs['original_fps']
        
        print(f"Dataset contains {num_sequences} sequences")
        print(f"FPS: {fps} (original: {original_fps})")


        #np.random.seed(42)
        #perm = np.random.permutation(num_sequences)
        
        #current_sequence_name = ['s01-box_grab_01', 's01-box_use_01', 's01-box_use_02', 's01-capsulemachine_grab_01', 's01-capsulemachine_use_01']

        # for smallish datasets, use single-thread parsing
        # for sample in list(perm[:int(0.9*num_sequences)]):
        #     with h5py.File(path, 'r') as f:
        #         sequence_name = f['sequence_names'][f'seq_{sample}'][:][0].decode('utf-8')
        #     if sequence_name in current_sequence_name:
        #         yield _parse_example(int(sample), path)

        # for sample in list(perm[:int(0.9*num_sequences)]):
        #     yield _parse_example(int(sample), path)
        
        for sample in list(range(num_sequences)):
            yield _parse_example(int(sample), path)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

