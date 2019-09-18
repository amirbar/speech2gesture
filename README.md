# Learning Individual Styles of Conversational Gestures
#### [Shiry Ginosar](http://people.eecs.berkeley.edu/~shiry)* , [Amir Bar](http://amirbar.net)* , Gefen Kohavi, [Caroline Chan](https://www.csail.mit.edu/person/caroline-chan), [Andrew Owens](http://andrewowens.com/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)
![alt text](data/teaser_gan_oliver_041.png "")
##### Back to [main project page](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html)

## Prerequisites:
1. `python 2.7`
2. `cuda 9.0`
3. `cuDNN v7.6.2`
4. `sudo apt-get install ffmpeg`
5. `pip install -r requirments.txt`

## Data
1. Download the dataset as described [here](data/dataset.md)

## Instructions
1. Extract training/validation data
2. Train a model
3. Perform inference using a trained model


### Extract training data
Start by extracting training data:
```
python -m data.train_test_data_extraction.extract_data_for_training --base_dataset_path <base_path> --speaker <speaker_name> -np <number of processes> --speaker <speaker name>`
```

```
once done you should see the following directories structure:
(notice train.csv and a train folder within the relevant speaker)

Gestures
├── frames.csv
├── train.csv
├── almaram
    ├── frames
    ├── videos
    ├── keypoints_all
    ├── keypoints_simple
    ├── videos
    └── train
...
└── shelly
    ├── frames
    ├── videos
    ├── keypoints_all
    ├── keypoints_simple
    ├── videos
    └── train
```

`train.csv` is a csv file in which every row represents a single training sample. Unlike in `frames.csv`, here,  a sample is few seconds long.  
![alt text](data/train.png "")

#### Columns documentation:
```
audio_fn - path to audio filename associated with training sample
dataset - train/dev/test
start - start time in the video
end - end time in the video
pose_fn - path to .npz file containing training sample
speaker - name of a speaker in the dataset
video_fn - name of the video file
```



### Training a speaker specific model
Training run command example:
```
python -m audio_to_multiple_pose_gan.train --gans 1 --name test_run --arch_g audio_to_pose_gans --arch_d pose_D --speaker oliver --output_path /tmp
```

During training, example outputs are saved in the define `output_path`


### Inference
optionally get a pretrained model [here](https://drive.google.com/drive/folders/1qvvnfGwas8DUBrwD4DoBnvj8anjSLldZ). 

Perform inference on a random sample from validation set:
```
python -m audio_to_multiple_pose_gan.predict_to_videos --train_csv <path/to/train.csv>--seq_len 64 --output_path </tmp/my_output_folder> --checkpoint <model checkpoint path> --speaker <speaker_name> -ag audio_to_pose_gans --gans 1
```
Perform inference on an audio sample:
```
python -m audio_to_multiple_pose_gan.predict_audio --audio_path <path_to_file.wav> --output_path </tmp/my_output_folder> --checkpoint <model checkpoint path> --speaker <speaker_name> -ag audio_to_pose_gans --gans 1

```


### Reference
If you found this code useful, please cite the following paper:


```
@InProceedings{ginosar2019gestures,
  author={S. Ginosar and A. Bar and G. Kohavi and C. Chan and A. Owens and J. Malik},
  title = {Learning Individual Styles of Conversational Gesture},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)}
  publisher = {IEEE},
  year={2019},
  month=jun
}
```
