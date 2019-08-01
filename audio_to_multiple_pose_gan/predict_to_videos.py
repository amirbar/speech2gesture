import matplotlib
from audio_to_multiple_pose_gan.config import create_parser, get_config
from common.pose_logic_lib import translate_keypoints
matplotlib.use('Agg')
import os
import argparse
import pandas as pd
from audio_to_multiple_pose_gan.model import PoseGAN


def main(args):
    df = pd.read_csv(args.train_csv)
    df_dev = df[(df['dataset'] == args.dataset) & (df['speaker'] == args.speaker)]

    if args.sample is not None:
        df_dev = df_dev.sample(n=args.sample)

    cfg = get_config(args.config)

    pgan = PoseGAN(args, seq_len=args.seq_len)
    if args.checkpoint:
        pgan.restore(args.checkpoint, scope_list=['generator', 'discriminator'])
    else:
        print "No checkpoint provided."

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    keypoints1_list, keypoints2_list, loss = pgan.predict_df(df_dev, cfg, [0,0], [0,0])
    keypoints1_list = translate_keypoints(keypoints1_list, [900, 290])
    keypoints2_list = translate_keypoints(keypoints2_list, [1900, 280])

    pgan.save_prediction_video_by_percentiles(df_dev, keypoints1_list, keypoints2_list, args.output_path, loss=loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser = create_parser(parser)
    parser.add_argument('-output_path', '--output_path', type=str, default='/tmp')
    parser.add_argument('-lb', '--loss_percentile_bgt', type=float, default=None)
    parser.add_argument('-ls', '--loss_percentile_smt', type=float, default=5)
    parser.add_argument('-dataset', '--dataset', type=str, default='dev')
    args = parser.parse_args()
    args.mode = 'inference'
    main(args)
