import librosa
from logging import getLogger

from common.consts import SR
from common.mel_features import log_mel_spectrogram

logger = getLogger(__name__)

RAW = 'raw'
LOG_MEL_SPECT = 'log_mel_spect'


def raw_repr(path, sr=None):
    wav, sr = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def log_mel_spectograms(path, audio_sample_rate=SR, log_offset=0.01, window_length_secs=0.025, hop_length_secs=0.010, num_mel_bins=64, num_min_hz=125,
                    num_max_hz=7500):
    if isinstance(path, str):
        wav = raw_repr(path, audio_sample_rate)
    else:
        wav = path
    return log_mel_spectrogram(wav, audio_sample_rate=audio_sample_rate, log_offset=log_offset,
                               window_length_secs=window_length_secs, hop_length_secs=hop_length_secs,
                               num_mel_bins=num_mel_bins, lower_edge_hertz=num_min_hz, upper_edge_hertz=num_max_hz)


repr_map = {
    RAW: raw_repr,
    LOG_MEL_SPECT: log_mel_spectograms
}


def get_repr(repr_name):
    return repr_map[repr_name]
