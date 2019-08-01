import random
import numpy as np
import os


def set_seeds(seed=1337):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def get_pose_path(base_path, speaker, pose_fn):
    return os.path.join(base_path, speaker, 'keypoints_simple', pose_fn)


def get_frame_path(base_path, speaker, frame_fn):
    return os.path.join(base_path, speaker, 'frames', frame_fn)


def get_video_path(base_path, speaker, video_fn):
    return os.path.join(base_path, speaker, 'videos', video_fn)
