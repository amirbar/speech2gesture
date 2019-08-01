import numpy as np
from common.consts import SPEAKERS_CONFIG


def normalize_relative_keypoints(k, speaker):
    return (k - SPEAKERS_CONFIG[speaker]['mean']) / (SPEAKERS_CONFIG[speaker]['std'] + np.finfo(float).eps)


def preprocess_to_relative(k, reshape=True, num_keypoints=49):
    reshaped = k.reshape((-1, 2, num_keypoints))
    relative = reshaped - reshaped[:, :, 0:1]
    if reshape:
        return relative.reshape((-1, num_keypoints * 2))
    return relative


def preprocess_to_relative_tensor(k, reshape=True, num_keypoints=21):
    import tensorflow as tf
    reshaped = tf.reshape(k, (-1, k.shape[1], 2, num_keypoints))
    relative = reshaped - reshaped[:, :, :, 0:1]
    if reshape:
        return tf.reshape(relative, (-1, k.shape[1], num_keypoints * 2))
    return relative


def de_normalize_relative_keypoints(k, speaker, scale_to_jon):
    keypoints = (k * (SPEAKERS_CONFIG[speaker]['std'] + np.finfo(float).eps) + SPEAKERS_CONFIG[speaker]['mean'])
    if scale_to_jon:
        keypoints = SPEAKERS_CONFIG[speaker]['scale_factor'] * keypoints
    return keypoints


def decode_pose_normalized_keypoints(encoded_keypoints, shift, speaker, scale_to_jon=True):
    encoded_keypoints = np.reshape(encoded_keypoints, (-1, 2, 49))
    encoded_keypoints[:, :, 0] = 0.
    encoded_keypoints = np.reshape(encoded_keypoints, (-1, 98))
    denormalaized = de_normalize_relative_keypoints(encoded_keypoints, speaker, scale_to_jon)
    denormalaized = np.reshape(denormalaized, (-1, 2, 49))
    return translate_keypoints(denormalaized, shift)


def decode_pose_normalized_keypoints_no_scaling(encoded_keypoints, shift, speaker):
    return decode_pose_normalized_keypoints(encoded_keypoints, shift, speaker, scale_to_jon=False)


def translate_keypoints(keypoints, shift):
    return keypoints + np.reshape(shift, (1, 2, 1))


def delete_face_keypoints(k, axis=1):
    '''
    Deletes the two eyes and nose from a model_23 set of openpose keypoints

    :param k: one set of keypoints with shape [2, #num_keypoints]  
    '''
    return np.delete(k, [7, 8, 9], axis=axis)  # in model_23 of openpose 7 is nose, 8,9 are eyes


def get_pose(arr, remove_new_keypoints=True):
    keypoints_batch = arr['pose']
    if keypoints_batch.shape[2] > 49 and remove_new_keypoints:
        return delete_face_keypoints(keypoints_batch, axis=2)
    return keypoints_batch


def _get_num_timesteps(x):
    if hasattr(x, 'get_shape'):
        num_timesteps = x.get_shape()[1].value
    else:
        num_timesteps = x.shape[1]
    return num_timesteps


def get_sample_output_by_config(x, config):
    if 'train_ratio' in config:
        num_timesteps = _get_num_timesteps(x)
        return x[:, conditioned_timesteps(config, num_timesteps):]
    return x


def conditioned_timesteps(config, num_timesteps):
    return int(config["train_ratio"] * num_timesteps)
