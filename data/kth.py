import random
import os
import numpy as np
import cv2
import torch


class KTH(object):
    def __init__(self, train=True, data_root='../../data/kth', seq_len=20, image_size=64):
        self.path = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.train = train
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

        self.dirs = os.listdir(self.path)

        self.seed_set = False

    def get_sequence(self):
        list_f = [x for x in os.listdir(self.path)]

        while True:
            rand_folder = random.choice(list_f)
            path_to_file = self.path + '/' + rand_folder
            file_name = random.choice(os.listdir(path_to_file))
            path_to_video = path_to_file + '/' + file_name
            vidcap = cv2.VideoCapture(path_to_video)
            n_frames = vidcap.get(7)
            stacked_frames = []
            while vidcap.isOpened():
                frame_id = vidcap.get(1)  # current frame number
                ret, frame = vidcap.read()
                if not ret or len(stacked_frames) > (self.seq_len - 1):
                    break
                frame = frame / 255.0
                if rand_folder == 'running' or rand_folder == 'walking' or rand_folder == 'jogging':
                    if frame_id % 1 == 0 and frame_id > 5:
                        frame = cv2.resize(frame, dsize=(self.image_size, self.image_size),
                                           interpolation=cv2.INTER_AREA)
                        stacked_frames.append(frame)
                elif n_frames < 350:
                    if frame_id % 1 == 0 and frame_id > 5:
                        frame = cv2.resize(frame, dsize=(self.image_size, self.image_size),
                                           interpolation=cv2.INTER_AREA)
                        stacked_frames.append(frame)
                else:
                    if frame_id % 1 == 0 and frame_id > 10:
                        frame = cv2.resize(frame, dsize=(self.image_size, self.image_size),
                                           interpolation=cv2.INTER_AREA)
                        stacked_frames.append(frame)
            if len(stacked_frames) == self.seq_len:
                break

        stacked_frames = np.reshape(stacked_frames, newshape=(self.seq_len, self.image_size, self.image_size, 3))
        return stacked_frames

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return len(self.dirs)*36*5 # arbitrary
