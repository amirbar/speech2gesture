import numpy as np


def compute_pck(pred, gt, alpha=0.2):
    '''
    :param pred: predicted keypoints on NxMxK where N is number of samples, M is of shape 2, corresponding to X,Y and K is the number of keypoints to be evaluated on
    :param gt:  similarly
    :param alpha: parameters controlling the scale of the region around the image multiplied by the max(H,W) of the person in the image. We follow https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf and set it to 0.1
    :return: mean prediction score
    '''
    pck_radius = compute_pck_radius(gt, alpha)
    keypoint_overlap = (np.linalg.norm(np.transpose(gt-pred, [0, 2, 1]), axis=2) <= (pck_radius))
    return np.mean(keypoint_overlap, axis=1)


def compute_pck_radius(gt, alpha):
    width = np.abs(np.max(gt[:, 0:1], axis=2) - np.min(gt[:, 0:1], axis=2))
    height = np.abs(np.max(gt[:, 1:2], axis=2) - np.min(gt[:, 1:2], axis=2))
    max_axis = np.concatenate([width, height], axis=1).max(axis=1)
    max_axis_per_keypoint = np.tile(np.expand_dims(max_axis, -1), [1, 48])
    return max_axis_per_keypoint * alpha
