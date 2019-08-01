import threading
from functools import partial
import numpy as np
from keras.utils.data_utils import GeneratorEnqueuer
from common.pose_logic_lib import normalize_relative_keypoints, preprocess_to_relative, \
    decode_pose_normalized_keypoints, get_pose, decode_pose_normalized_keypoints_no_scaling
import logging
logging.basicConfig()
from logging import getLogger

logger = getLogger(__name__)


def load_train(process_row, batch_size, df, batch_generator, in_memory=False, workers=7, max_queue_size=128, use_multiprocessing=True):
    train_generator, num_samples = load_set('train', process_row, batch_size, df, batch_generator=batch_generator, is_train=True, in_memory=in_memory)
    enqueuer = GeneratorEnqueuer(train_generator, use_multiprocessing=use_multiprocessing)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    output_generator = enqueuer.get()
    return output_generator , num_samples


def load_set(set_name, process_npz_row, batch_size, df, batch_generator, is_train=True, in_memory=False):
    df = df[df['dataset'] == set_name]
    return set_generator(df, is_train, generate_batch=batch_generator, batch_size=batch_size, process_row=process_npz_row, in_memory=in_memory), len(df)


def get_processor(config):
    processing_type = config["processor"]
    f = audio_pose_mel_spect
    if processing_type == 'audio_to_pose':
        d = decode_pose_normalized_keypoints
    elif processing_type == 'audio_to_pose_inference':
        d = decode_pose_normalized_keypoints_no_scaling
    else:
        raise ValueError("Wrong Processor")
    return partial(f, config), d


def preprocess_x(x, config):
    if len(x) > config['input_shape'][1]:
        x = x[:config['input_shape'][1]]
    elif len(x) < config['input_shape'][1]:
        x = np.pad(x, [0, config['input_shape'][1] - len(x)], mode='constant', constant_values=0)
    return x


def audio_pose_mel_spect(config, row):
    if "audio" in row:
        x = row["audio"]
    else:
        arr = np.load(row['pose_fn'])
        x = arr["audio"]
    x = preprocess_x(x, config)
    y = preprocess_to_relative(get_pose(arr))
    y = normalize_relative_keypoints(y, row['speaker'])
    if "flatten" in config and config["flatten"]:
        y = y.flatten()
    return x, y


def generate_batch(df, process_row, batch_size):
    X, Y = [], []
    while len(X) < batch_size:
        row = df.sample(n=1).iloc[0]
        x_sample, y_sample = process_row(row)
        X.append(x_sample)
        Y.append(y_sample)
    Y = np.array(Y)
    X = np.array(X)
    return X, Y


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        assert self.lock is not bcolz_lock

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.

    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def set_generator(df, random_choice, batch_size, process_row, generate_batch, raise_e=False, in_memory=False):
    if in_memory:
        X, Y = generate_batch(df, process_row, batch_size)
        yield X, Y
        raise StopIteration()

    if random_choice:
        while True:
            X, Y = generate_batch(df, process_row, batch_size)
            yield X, Y
            # yield np.random.normal(0,1, size=(128, 256, 256, 1)), np.random.normal(0, 1, size=(128, 18))
    else:
        while True:
            for i in range(0, len(df), batch_size):
                rows = df[i:i+batch_size]
                X, Y = generate_batch(rows, process_row, batch_size)
                yield X, Y
            if raise_e:
                raise StopIteration()


bcolz_lock = threading.Lock()