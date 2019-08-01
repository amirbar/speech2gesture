import matplotlib
matplotlib.use('Agg')
import logging
logging.basicConfig()
import argparse
from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
from common.audio_lib import get_img_pos_wav, get_interval_start_end
from tqdm import tqdm
from logging import getLogger
logger = getLogger("train.saver")
# logger.setLevel(logging.ERROR)
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-c','--csv_path', default='/data/efros/dataset/Gestures/frames_df_2_16_19.csv')
parser.add_argument('-sp','--save_path', default='/data/efros/dataset/Gestures_extras/test_256_all/') # 1 to save 0 to skip audio
parser.add_argument('-sc','--save_csv', default='/data/efros/dataset/Gestures_extras/test_256_all.csv')
parser.add_argument('-b','--base_dataset_path', default='/data/efros/dataset/Gestures/')
parser.add_argument('-nc','--num_cores', type=int, default=1)
parser.add_argument('-nf','--num_frames', type=int, default=64)
parser.add_argument('-d','--debug', type=int, default=0)
parser.add_argument('-s','--speaker', default=None)
parser.add_argument('-sa','--sample', default=1000, type=int)

args = parser.parse_args()

num_frames = args.num_frames
audio_base_path = os.path.join(args.save_path, '%s/audio/%s-%s-%s.wav')
frames_base_path = os.path.join(args.save_path, '%s/npz_pose/%s-%s-%s.npz')
FPS = 30./2
for p in [os.path.dirname(audio_base_path), os.path.dirname(frames_base_path)]:
    if not(os.path.exists(p)):
        os.makedirs(p)

def main():
    df = pd.read_csv(args.csv_path)
    if args.speaker is not None:
        df = df[df['speaker'] == args.speaker]
    df = df[df['dataset'] == 'test']

    df['ones'] = 1
    grouped = df.groupby('interval_id').agg({'ones': sum}).reset_index()
    grouped = grouped[grouped['ones'] >= num_frames][['interval_id']]
    df = df.merge(grouped, on='interval_id')

    print "Number of frames: %s"%(len(df))


    dfs = [df[df['speaker'] == speaker] for speaker in df['speaker'].unique()]
    del df
    if args.num_cores > 1:
        p = Pool(args.num_cores)
        dfs = p.map(save_video_samples, dfs)
    else:
        dfs = map(save_video_samples, dfs)
    pd.concat(dfs).to_csv(args.save_csv, index=False)


def save_video_samples(df):
    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [], 'pose_fn': [], 'audio_fn': [], 'video_fn': [], 'speaker': []}
    intervals = df['interval_id'].unique()
    i=0
    pbar = tqdm(total=args.sample)
    while i<args.sample:

        try:
            interval = intervals[np.random.randint(0, len(intervals))]
            df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
            video_fn = df_interval.iloc[0]['video_fn']
            speaker_name = df_interval.iloc[0]['speaker']
            if len(df_interval) < num_frames:
                continue

            idx = np.random.randint(0, len(df_interval) - num_frames + 1)
            sample = df_interval[idx:idx + num_frames]

            interval_start, interval_end = get_interval_start_end(sample)
            audio_out_path = audio_base_path % (speaker_name, interval, interval_start, interval_end)
            img_fns, poses, wav = get_img_pos_wav(sample, video_fn, audio_out_path, args.base_dataset_path, SR)

            frames_out_path = frames_base_path % (speaker_name, interval, sample.iloc[0]['pose_dt'], sample.iloc[-1]['pose_dt'])

            if not (os.path.exists(os.path.dirname(frames_out_path))):
                os.makedirs(os.path.dirname(frames_out_path))

            np.savez(frames_out_path, pose=poses, imgs=img_fns, audio=wav)

            data_dict["dataset"].append(df_interval.iloc[0]['dataset'])
            data_dict["start"].append(sample.iloc[0]['pose_dt'])
            data_dict["end"].append(sample.iloc[-1]['pose_dt'])
            data_dict["interval_id"].append(interval)
            data_dict["pose_fn"].append(frames_out_path)
            data_dict["audio_fn"].append(audio_out_path)
            data_dict["video_fn"].append(video_fn)
            data_dict["speaker"].append(speaker_name)
            i+=1
            pbar.update(1)
        except Exception as e:
            logger.exception(e)
            continue

    pbar.close()

    return pd.DataFrame.from_dict(data_dict)


if __name__ == "__main__":
    main()
