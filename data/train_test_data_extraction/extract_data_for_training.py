import matplotlib
matplotlib.use('Agg')
from common.consts import FRAMES_PER_SAMPLE, SR
from common.utils import get_pose_path, get_frame_path, get_video_path
import argparse
from multiprocessing import Pool
from common.audio_repr import raw_repr
import os
import numpy as np
import pandas as pd
from common.audio_lib import save_audio_sample_from_video
from tqdm import tqdm
from logging import getLogger
import logging
logging.basicConfig()

logger = getLogger(__name__)

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-b','--base_dataset_path', help="dataset root path")
parser.add_argument('-np','--num_processes', type=int, default=1)
parser.add_argument('-nf','--num_frames', type=int, default=FRAMES_PER_SAMPLE)
parser.add_argument('-s','--speaker', default=None)


args = parser.parse_args()

num_frames = args.num_frames
AUDIO_FN_TEMPLATE = os.path.join(args.base_dataset_path, '%s/train/audio/%s-%s-%s.wav')
TRAINING_SAMPLE_FN_TEMPLATE = os.path.join(args.base_dataset_path, '%s/train/npz/%s-%s-%s.npz')


def save_video_samples(df):
    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [], 'pose_fn': [], 'audio_fn': [], 'video_fn': [], 'speaker': []}
    intervals = df['interval_id'].unique()
    for interval in tqdm(intervals):
        try:
            df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
            video_fn = df_interval.iloc[0]['video_fn']
            speaker_name = df_interval.iloc[0]['speaker']
            if len(df_interval) < num_frames:
                logger.warning("interval: %s, num frames: %s. skipped"%(interval, len(df_interval)))
                continue
            poses = np.array([np.loadtxt(get_pose_path(args.base_dataset_path, row['speaker'], row['pose_fn'])) for _, row in df_interval.iterrows()])
            img_fns = np.array([get_frame_path(args.base_dataset_path, row['speaker'], row['frame_fn']) for _, row in df_interval.iterrows()])

            interval_start = str(pd.to_datetime(df_interval.iloc[0]['pose_dt']).time())
            interval_end = str(pd.to_datetime(df_interval.iloc[-1]['pose_dt']).time())

            audio_out_path = AUDIO_FN_TEMPLATE % (speaker_name, interval, interval_start, interval_end)
            save_audio_sample_from_video(get_video_path(args.base_dataset_path, speaker_name, video_fn), audio_out_path, interval_start, interval_end)

            interval_start = pd.to_timedelta(df_interval.iloc[0]['pose_dt'])

            interval_audio_wav, sr = raw_repr(audio_out_path, SR)
            for idx in range(0, len(df_interval) - num_frames, 5):

                sample = df_interval[idx:idx + num_frames]
                start = (pd.to_timedelta(sample.iloc[0]['pose_dt'])-interval_start).total_seconds()*SR
                end = (pd.to_timedelta(sample.iloc[-1]['pose_dt'])-interval_start).total_seconds()*SR
                frames_out_path = TRAINING_SAMPLE_FN_TEMPLATE % (speaker_name, interval, sample.iloc[0]['pose_dt'], sample.iloc[-1]['pose_dt'])
                wav = interval_audio_wav[int(start): int(end)]

                if not (os.path.exists(os.path.dirname(frames_out_path))):
                    os.makedirs(os.path.dirname(frames_out_path))

                np.savez(frames_out_path, pose=poses[idx:idx + num_frames], imgs=img_fns[idx:idx + num_frames], audio=wav)

                data_dict["dataset"].append(df_interval.iloc[0]['dataset'])
                data_dict["start"].append(sample.iloc[0]['pose_dt'])
                data_dict["end"].append(sample.iloc[-1]['pose_dt'])
                data_dict["interval_id"].append(interval)
                data_dict["pose_fn"].append(frames_out_path)
                data_dict["audio_fn"].append(audio_out_path)
                data_dict["video_fn"].append(video_fn)
                data_dict["speaker"].append(speaker_name)
        except Exception as e:
            logger.exception(e)
            continue
    return pd.DataFrame.from_dict(data_dict)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(args.base_dataset_path, "frames_df_10_19_19.csv"))
    if args.speaker is not None:
        df = df[df['speaker'] == args.speaker]
    df = df[(df['dataset'] == 'train') | (df['dataset'] == 'dev')]


    intervals_unique = df['interval_id'].unique()
    print "Number of unique intervals to save: %s" % (len(intervals_unique))


    intervals = np.array_split(intervals_unique, args.num_processes)
    dfs = [df[df['interval_id'].isin(interval)] for interval in intervals]
    del df
    if args.num_processes > 1:
        p = Pool(args.num_processes)
        dfs = p.map(save_video_samples, dfs)
    else:
        dfs = map(save_video_samples, dfs)
    pd.concat(dfs).to_csv(os.path.join(args.base_dataset_path, "train.csv"), index=False)
