import argparse
from tqdm import tqdm
import subprocess
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
parser.add_argument('-output_path', '--output_path', default='output directory to save cropped intervals', required=True)
parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

args = parser.parse_args()


def save_interval(input_fn, start, end, output_fn):
    cmd = 'ffmpeg -i "%s" -ss %s -to %s -strict -2 "%s" -y' % (
    input_fn, start, end, output_fn)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    df_intervals = pd.read_csv(os.path.join(args.base_path, 'intervals_df.csv'))
    if args.speaker:
        df_intervals = df_intervals[df_intervals["speaker"] == args.speaker]

    for _, interval in tqdm(df_intervals.iterrows(), total=len(df_intervals)):
        try:
            start_time = str(pd.to_datetime(interval['start_time']).time())
            end_time = str(pd.to_datetime(interval['end_time']).time())
            input_fn = os.path.join(args.base_path, interval['speaker'], "videos", interval["video_fn"])
            output_fn = os.path.join(args.output_path, "%s_%s_%s-%s.mp4"%(interval["speaker"], interval["video_fn"], str(start_time), str(end_time)))
            print(input_fn, output_fn)
            save_interval(input_fn, str(start_time), str(end_time), output_fn)
        except Exception as e:
            print(e)
            print("couldn't crop interval: %s"%interval)
