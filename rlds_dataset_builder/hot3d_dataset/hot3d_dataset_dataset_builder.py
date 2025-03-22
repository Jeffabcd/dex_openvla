from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import cv2

class Hot3dDataset(tfds.core.GeneratorBasedBuilder):
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
                            shape=(276,),
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
                        shape=(276,),
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
            'train': self._generate_examples(path=f'../../hot3d/hot3d/preprocess_dataset/{dataset_name}/*/*.hdf5', split='train'),
            #'val': self._generate_examples(path='../../hot3d/hot3d/preprocess_dataset/{dataset_name}/*/*.hdf5', split='val'),
            #'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path, split='') -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            with h5py.File(episode_path, 'r') as hdf_file:
                images = hdf_file['images'][:]
                wrist_states = hdf_file['wrist_states'][:]
                wrist_actions = hdf_file['wrist_actions'][:]
                hand_states = hdf_file['hand_states'][:]
                hand_actions = hdf_file['hand_actions'][:]
                lang = hdf_file['language_instruction'][()].decode('utf-8')
                sequence_name = hdf_file['sequence_name'][()].decode('utf-8')
                demo_timestamps = hdf_file['demo_timestamps'][:]
            #print(left_pos.shape, right_pos.shape)
            #print(left_a.shape, right_a.shape)
            #print(lang)
            if wrist_states.shape[0] != wrist_actions.shape[0]:
                print(f'{episode_path} dimension mismatch!')
                raise

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, _image in enumerate(images):
                # compute Kona language embedding
                language_embedding = self._embed([lang])[0].numpy()

                rotated_image = cv2.rotate(_image, cv2.ROTATE_90_CLOCKWISE)
                
                pre_image = cv2.resize(rotated_image, (256, 256), interpolation=cv2.INTER_AREA)

                episode.append({
                    'observation': {
                        'image': pre_image,
                        'wrist_state': wrist_states[i].astype(np.float32),
                        'hand_state': hand_states[i].astype(np.float32),
                    },
                    'wrist_action': wrist_actions[i].astype(np.float32), 
                    'hand_action': hand_actions[i].astype(np.float32), 
                    'action': wrist_actions[i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(wrist_states) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(wrist_states) - 1),
                    'is_terminal': i == (len(wrist_states) - 1),
                    'language_instruction': lang,
                    'sequence_name': sequence_name,
                    'timestamp_ns': demo_timestamps[i],
                    'next_timestamp_ns': demo_timestamps[i+1] if i < (len(wrist_states)-1) else demo_timestamps[i],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)
        #print('##############', episode_paths)
        if split == 'train':
           # episode_paths = episode_paths[:1]
            #episode_paths = episode_paths[:int(0.9* len(episode_paths))]
            episode_paths = episode_paths
        elif split == 'val':
            #episode_paths = episode_paths[-1:]
            episode_paths = episode_paths[int(0.9* len(episode_paths)):]

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

