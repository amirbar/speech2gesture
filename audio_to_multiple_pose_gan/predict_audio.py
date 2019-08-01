import matplotlib

from common.pose_logic_lib import translate_keypoints

matplotlib.use('Agg')
import numpy as np
from common.audio_repr import raw_repr
from common.consts import SR
from audio_to_multiple_pose_gan.config import create_parser, get_config
from common.pose_plot_lib import save_video_from_audio_video, save_side_by_side_video
import os
import argparse
from audio_to_multiple_pose_gan.model import PoseGAN

def main(args):

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    audio_fn = args.audio_path
    audio, _ = raw_repr(audio_fn, SR)
    pose_shape = int(15 * float(audio.shape[0]) / SR)
    padded_pose_shape = pose_shape + (2**5) - pose_shape%(2**5)
    padded_audio_shape = padded_pose_shape * SR / 15
    padded_audio = np.pad(audio, [0, padded_audio_shape - audio.shape[0]], mode='reflect')

    cfg = get_config(args.config)
    pgan = PoseGAN(args, seq_len=padded_pose_shape)
    if args.checkpoint:
        pgan.restore(args.checkpoint, scope_list=['generator', 'discriminator'])
    else:
        print "No checkpoint provided."

    padded_pred_kpts = pgan.predict_audio(padded_audio, cfg, args.speaker, [0,0])
    padded_pred_kpts = translate_keypoints(padded_pred_kpts, [900, 290])

    pred_kpts = padded_pred_kpts[:pose_shape]

    tmp_output_dir = '/tmp'
    mute = os.path.join(tmp_output_dir, 'mute_pred.mp4')
    output_fn = os.path.join(args.output_path, 'output.mp4')
    save_side_by_side_video(tmp_output_dir, pred_kpts, pred_kpts, mute, delete_tmp=False)
    save_video_from_audio_video(audio_fn, mute, output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser = create_parser(parser)
    parser.add_argument('-audio_path', '--audio_path', type=str, required=True)
    args = parser.parse_args()
    args.mode = 'inference'
    main(args)
