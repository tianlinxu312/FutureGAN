import random
import os
import numpy as np
import socket
import torch
from scipy import misc

import collections
import tensorflow as tf
nest = tf.nest

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 7,
        }



"""Minimal data reader for GQN TFRecord datasets."""

# nest = tf.contrib.framework.nest
nest = tf.nest
seed = 1

DatasetInfo = collections.namedtuple('DatasetInfo', ['basepath', 'train_size', 'test_size', 'frame_size',
                                                     'sequence_size'])
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


class DataReader(object):
    """Minimal queue based TFRecord reader.
      You can use this reader to load the datasets used to train Generative Query
      Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
      See README.md for a description of the datasets and an example of how to use
      the reader.
      """

    def __init__(self,
                 dataset,
                 time_steps,
                 root,
                 mode='train',
                 # Optionally reshape frames
                 custom_frame_size=None):
        """Instantiates a DataReader object and sets up queues for data reading.
        Args:
          dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
              'rooms_free_camera_no_object_rotations',
              'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
              'shepard_metzler_7_parts'].
          time_steps: integer, number of views to be used to assemble the context.
          root: string, path to the root folder of the data.
          mode: (optional) string, one of ['train', 'test'].
          custom_frame_size: (optional) integer, required size of the returned
              frames, defaults to None.
          num_threads: (optional) integer, number of threads used to feed the reader
              queues, defaults to 4.
          capacity: (optional) integer, capacity of the underlying
              RandomShuffleQueue, defualts to 256.
          min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
              RandomShuffleQueue, defualts to 128.
          seed: (optional) integer, seed for the random number generators used in
              the reader.
        Raises:
          ValueError: if the required version does not exist; if the required mode
             is not supported; if the requested time_steps is bigger than the
             maximum supported for the given dataset version.
        """

        if dataset not in _DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                             'are {}'.format(dataset, _DATASETS.keys()))

        if mode not in _MODES:
            raise ValueError('Unsupported mode {} requested. Supported modes '
                             'are {}'.format(mode, _MODES))

        self._dataset_info = _DATASETS[dataset]

        if time_steps > self._dataset_info.sequence_size:
            raise ValueError(
                'Maximum support context size for dataset {} is {}, but '
                'was {}.'.format(
                    dataset, self._dataset_info.sequence_size, time_steps))

        self.time_steps = time_steps
        self._custom_frame_size = custom_frame_size

        with tf.device('/cpu'):
            self._queue = _get_dataset_files(self._dataset_info, mode, root)

    def get_dataset_from_path(self, buffer=100):
        read_data = tf.data.Dataset.list_files(self._queue)
        dataset = read_data.repeat().shuffle(buffer_size=buffer)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        return dataset

    def provide_dataset(self):
        """Instantiates the ops used to read and parse the data into tensors."""
        def read_tfrecord(serialized_example):
            feature_map = {'frames': tf.io.FixedLenFeature(shape=self._dataset_info.sequence_size, dtype=tf.string)}
            example = tf.io.parse_example(serialized_example, feature_map)
            frames = self._preprocess_frames(example)
            return frames

        dataset = self.get_dataset_from_path()
        dataset = dataset.map(read_tfrecord, num_parallel_calls=4)
        return dataset

    def _preprocess_frames(self, example):
        """Instantiates the ops used to preprocess the frames data."""
        frames = tf.concat(example['frames'], axis=0)
        frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(_convert_frame_data,  tf.reshape(frames, [-1]),
                                                                   dtype=tf.float32))
        dataset_image_dimensions = tuple([self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
        frames = tf.reshape(frames, (-1, self._dataset_info.sequence_size) + dataset_image_dimensions)
        if (self._custom_frame_size and
                self._custom_frame_size != self._dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
            frames = tf.image.resize(frames, new_frame_dimensions[:2])
            frames = tf.reshape(frames, (-1, self._dataset_info.sequence_size) + new_frame_dimensions)

        frames = tf.transpose(tf.squeeze(frames[:, :self.time_steps, :, :]), (1, 0, 2, 3))
        frames = tf.transpose(frames, (0, 1, 3, 2))
        frames = tf.transpose(frames, (0, 2, 1, 3))
        frames = tf.transpose(frames, (1, 0, 2, 3))
        return tf.cast(frames, tf.float32)
