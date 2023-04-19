import gzip
import itertools
import json
import os
import pickle
from math import ceil

import h5py
import keras
import numpy
import sidekit
import soundfile
from _pickle import dump
from datasets import ASVspoof
from helpers import matrix_to_vector
from obspy.signal.util import enframe, next_pow_2
from preprocessing import split_frames
from python_speech_features.sigproc import preemphasis
from scipy.fftpack.realtransforms import dct
from sidekit.frontend.features import framing, power_spectrum, trfbank
from sidekit.frontend.vad import pre_emphasis


def load_features(feature_type, subset="", asarray=True, split=False):
    """
    Loads extracted features.

    Args:
        feature_type: Feature method to find filename.
        subset: Subset category of dataset.
        asarray: An option to return features as numpy array.
        split: Split option for frame wise classification.

    Returns:
        x_data, y_data, labels, indexes, batch_size and  file_list.
    """
    # Get config file to read feature directory.
    os.chdir("config")

    # Load file.
    with open("file_structure.json", "r") as fp:
        data = json.load(fp)
        feature_dir = data["project_path"] + data["data"]["feature"]

    # Check if feature folder exist.
    try:
        os.path.isdir(feature_dir)
    except FileNotFoundError as e:
        assertion_text = "\nFeature folder not found."
        raise AssertionError(assertion_text) from e
    path_to_feats = feature_dir + feature_type + "/" + subset
    os.chdir("code")

    # Load data.
    labels = []
    x_data = []
    files = os.listdir(path_to_feats)
    for file in files:
        with open(f"{path_to_feats}/{file}", "rb+") as fp:
            pickled_data = pickle.load(fp)
            x_data.append(pickled_data[0])
            labels.append(pickled_data[1])
        fp.close()

    # Split features for frame-wise classification.
    if split is True:
        x_data, y_data, indexes_ = split_frames(x_data, labels)
    else:
        indexes_ = None
        y_data = labels
    if asarray is True:
        # Convert to numpy array.
        x_data = numpy.vstack(x_data).astype(
            "float32")  # numpy.asarray(x_data)

    # Represent labels in categorical form.
    # noinspection PyUnresolvedReferences
    y_data = keras.utils.to_categorical(numpy.asarray(y_data), 2)

    # Determine min batch size.
    batch_size = len(x_data) // 100 if len(x_data) > 10000 else len(x_data)
    # File list.
    file_list = ASVspoof(subset).file_list

    # Check ""nan" value existence and return data.
    """
    any_nan = numpy.isnan(x_data).any()
    if any_nan is True or any_nan is numpy.bool_(True):
        raise AssertionError(""nan" value occurred.")
    """
    return x_data, y_data, labels, indexes_, batch_size, file_list


def fix_features(method, feats):
    """
    Fixes given features according to methods "1", "2", "3". 1" for cropping
    every data to min value, "2" for scaling data to mean value, "3" for
    splitting matrices to vector of all data rows.

    Args:
        method: Fixing method that can be "1", "2", "3".
        feats: A data list that contains numpy.ndarray.

    Returns:
        Formatted data according to a method.
    """
    fixed_feats = []
    if method == "1":
        # Find length of feats.
        feat_lengths = numpy.zeros([len(feats)], dtype=int)
        for i in range(len(feats)):
            feat_lengths[i] = len(feats[i])
        min_len = feat_lengths.min  # ()

        # Crop every row that has longer than minimum length.
        for i in range(len(feats)):
            fixed_feats.append(matrix_to_vector(feats[i]))
            fixed_feats[i] = fixed_feats[i][:min_len]
    elif method == "2":
        feat_lengths = [len(feats[i]) for i in range(len(feats))]
        mean_of_len = ceil(sum(feat_lengths) / len(feats))

        # Find ceps number.
        ceps_number = len(feats[0][0])

        # Scale every row to mean value of length by cropping longer and
        # adding values(from first values) to smaller ones.
        for i in range(len(feats)):
            buffer = numpy.zeros([mean_of_len, ceps_number], dtype=float)

            # Find how many missing value in feats.
            missing = len(feats[i]) - mean_of_len

            # Keep relevant feat.
            feat = feats[i]

            # If there are fewer values.
            if missing < 0:
                # Make it positive.
                missing = (-1) * missing

                # Fill with first values.
                for j in range(ceps_number):
                    buffer[j, 0:missing] = feat[j, 0:missing]

                # Get another values.
                buffer[missing:] = feat[:]
            elif missing > 0:  # If there are more values.
                # Fill with first values.
                for j in range(ceps_number):
                    buffer[j, :] = feat[j, 0:mean_of_len]
            fixed_feats.append(buffer)
    elif method == "3":
        # Convert matrix to vector.
        for i in range(len(feats)):
            buffer = feats[i]
            buffer = matrix_to_vector(buffer)
            fixed_feats.append(buffer)
    else:
        print("Option does not exist. \n")

    return fixed_feats


def feature_normalization(feat):
    """
    Normalizes given features.

    Args:
        feat: Input data.

    Returns:
        Normalized data.
    """
    mu = numpy.mean(feat)
    sigma = numpy.std(feat)
    if len(feat.shape) == 1:
        normalized_feat = numpy.zeros([len(feat)], dtype=float)
        for i in range(len(feat)):
            normalized_feat[i] = (feat[i] - mu) / sigma
    else:
        normalized_feat = numpy.zeros([feat.shape[0], feat.shape[1]],
                                      dtype=float)
        for i, j in itertools.product(range(feat.shape[0]), range(feat.shape[1])):
            normalized_feat[i][j] = (feat[i][j] - mu) / sigma

    return normalized_feat


# noinspection DuplicatedCode
def long_term_spectra(x, fs):
    """
    Extracts long-term spectral statistics of given signal. For more details on
    implementation this method: See Section III of "Presentation Attack
    Detection  Using Long-Term Spectral Statistics for Trustworthy Speaker
    Verification"

    Args:
        x: Pure speech signal(note that pre-emphasis, windowing etc. applied
        to signal inside the function).
        fs: Sample rate of the signal.

    Returns:
        LTAS features.
    """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Pre-emphasis on signal.
    x = preemphasis(x, 0.97)  # x = sidekit.frontend.vad.pre_emphasis(x, 0.97)

    # Calculate frame and shift length.
    frame_len = 20 * fs / 1000
    frame_step = frame_len / 2

    # Apply hamming window and split signal into frames.
    frames, win_len, no_win = enframe(x + eps, numpy.hamming(frame_len),
                                      int(frame_step))

    # Number of fft points.
    n_fft = next_pow_2(frame_len)

    # A place holder for ltas.
    ltas = numpy.zeros([no_win, int(n_fft / 2) + 1], dtype=float)

    # N point fft of signal.
    for i in range(no_win):
        y = frames[i, :]
        y = abs(numpy.fft.fft(y, n_fft))
        ltas[i, :] = y[1:int(n_fft / 2) + 2]

    # Declare mu, sigma and feats.
    mu = numpy.zeros([int(n_fft / 2)], dtype=float)
    sigma = numpy.zeros([int(n_fft / 2)], dtype=float)
    feats = numpy.zeros([n_fft], dtype=float)

    # Feats: Mean and standard deviation of ltas.
    for i in range(int(n_fft / 2)):
        ltas_log = numpy.log([ltas[:, i] + eps])
        mu[i] = numpy.mean(ltas_log)
        sigma[i] = numpy.std(ltas_log)
        feats[i + int(n_fft / 2)] = sigma[i]
        feats[i] = mu[i]

    return feats


def ltas(input_sig, fs=16000, fc=0, win_time=0.02, shift_time=0.01):
    """
    Extracts long-term spectral statistics of given speech signal.

    Args:
        input_sig: Speech signal.
        fs: Sample rate of the signal.
        fc: F cut frequency.
        win_time: Length of the sliding window in seconds.
        shift_time: Shift between two analyses.

    Returns:
        Mean and variance statistics of fourier magnitude spectrum.
    """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Pre-emphasis on signal.
    input_sig = pre_emphasis(input_sig, 0.97)

    # Calculate frame and overlap length in terms of samples.
    window_len = int(round(win_time * fs))
    overlap = window_len - int(shift_time * fs)

    # Split signal into frames.
    framed_sig = framing(sig=input_sig, win_size=window_len,
                         win_shift=window_len - overlap,
                         context=(0, 0), pad="zeros")

    # Windowing.
    window = numpy.hamming(window_len)
    windowed_sig = framed_sig * window

    # Number of fft points.
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_len)))

    # Low frequency cutting variables.
    start = 0 if fc == 0 else int((fc * n_fft) / fs)
    end = int(n_fft / 2) + 1
    n_elements = int(end - start)

    # A placeholder for spectrum.
    spectrum = numpy.zeros([framed_sig.shape[0], n_elements])

    # N point FFT of signal.
    fft = numpy.fft.rfft(windowed_sig, n_fft)

    # Fourier magnitude spectrum computation.
    log_magnitude = numpy.log(abs(fft) + eps)

    # Crop components according to fc.
    for frame in range(windowed_sig.shape[0]):
        spectrum[frame] = log_magnitude[frame][start:end]

    # Concatenate mean and variance statistics.
    mu = numpy.mean(spectrum, axis=0)
    sigma = numpy.std(spectrum, axis=0)

    return numpy.concatenate((mu, sigma))


def extract_frames(input_sig, fs=16000, win_time=0.02, shift_time=0.01):
    """
    Splits signal up into (overlapping) frames beginning at increments
    of frame_step. Each frame is multiplied by the hamming window.

    Args:
        input_sig: Speech signal to split in frames.
        fs: Sample rate of signal.
        win_time: Length of the sliding window in seconds.
        shift_time: Shift between two analyses.

    Returns:
        Output matrix, each frame occupies one row.
    """
    # Pre-emphasis on signal.
    input_sig = pre_emphasis(input_sig, 0.97)

    # Calculate frame and shift length.
    window_len = int(round(win_time * fs))
    overlap = window_len - int(shift_time * fs)

    # Split signal into frames.
    framed_sig = framing(sig=input_sig,
                         win_size=window_len,
                         win_shift=window_len - overlap,
                         context=(0, 0), pad="zeros")

    # Windowing.
    window = numpy.hamming(window_len)
    return framed_sig * window


def mfcc(input_sig, lowfreq=100, maxfreq=8000, nlinfilt=0, nlogfilt=24,
         nwin=0.02, fs=16000, nceps=13, shift=0.01, get_spec=False,
         get_mspec=False, prefac=0.97):
    # Compute power spectrum
    spec, log_energy = power_spectrum(input_sig,
                                      fs,
                                      win_time=nwin,
                                      shift=shift,
                                      prefac=prefac)

    # Filter the spectrum through the triangle filter-bank.
    n_fft = 2 ** int(numpy.ceil(numpy.log2(int(round(nwin * fs)))))
    fbank = trfbank(fs, n_fft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]

    # A tester avec log10 et log.
    mspec = numpy.log(numpy.dot(spec, fbank.T) + 2.2204e-16)

    # Use the DCT to "compress" the coefficients (spectrum -> cepstrum domain)
    # The C0 term is removed as it is the constant term.
    ceps = dct(mspec, type=2, norm="ortho", axis=-1)[:, 1:nceps + 1]
    result = [ceps, log_energy]
    if get_spec:
        result.append(spec)
    else:
        result.append(None)
        del spec
    if get_mspec:
        result.append(mspec)
    else:
        result.append(None)
        del mspec

    return result


class Feature(object):
    """
    A class to extract features from speech signal using SIDEKIT package and
    some other functions created above. In addition, this class provide to
    read, write, pickle, unpickle, visualize and other utilities for
    manipulating extracted features.
    """

    def __init__(self, obj):
        """
        Initializes class for given param(dataset).

        Args:
            obj: Dataset object.
        """
        self.obj = obj
        self.x = []
        self.feats = []

    def sidekit_extractor(self, path_to_folder):
        """
        Extracts mfcc and saves to .h5 file according to parameters below.
        This method adapted from SIDEKIT tutorials.

        https://www-lium.univ-lemans.fr/sidekit/tutorial/featuresextractor.html
        """
        extractor = sidekit.FeaturesExtractor(
            audio_filename_structure=self.obj.path_to_wav + "{}.wav",
            feature_filename_structure=path_to_folder + "{}.h5",
            sampling_frequency=16000,
            lower_frequency=0,
            higher_frequency=8000,
            filter_bank="lin",
            filter_bank_size=30,
            window_size=0.02,
            shift=0.01,
            ceps_number=20,
            vad=None,
            snr=40,
            pre_emphasis=0.97,
            save_param=["cep"],
            keep_all_features=False)

        # Extract features for genuine and spoof for data.
        for i in range(len(self.obj.file_list)):
            extractor.save(self.obj.file_list[i])

    def extract(self, file_path, path_to_file, f_type, fs=16000, shift=0.01,
                d=1, p=3,
                k=7, left_ctx=12, right_ctx=12, win_time=0.025, f_n=True):
        # Read audio files and store into x list.
        for i in range(len(self.obj.file_list)):
            self.x.append(soundfile.read(
                self.obj.path_to_wav + self.obj.file_list[i] + ".wav")[0])

        # Extract selected method to data in x.
        for i in range(len(self.obj.file_list)):
            if "mfcc" in f_type:
                """
                mfcc = sidekit.frontend.features.mfcc(self.x[i],
                                                      lowfreq=lowfreq,
                                                      maxfreq=maxfreq,
                                                      nlinfilt=nlinfilt,
                                                      nlogfilt=nlogfilt,
                                                      nwin=nwin,
                                                      fs=fs,
                                                      nceps=nceps,
                                                      shift=shift,
                                                      get_spec=get_spec,
                                                      get_mspec=get_mspec,
                                                      prefac=prefac)[0]  
                """
                # self.obj.subset + "/"
                filename = self.obj.protocol[i, 0][:9]
                file = h5py.File((file_path + filename + ".h5"), "r+")
                # noinspection PyUnresolvedReferences
                feats = sidekit.frontend.io.read_dict_hdf5(path_to_file)
                file.close()
                mfcc_feats = feats[f"{filename}/cep"]

                if f_type == "mfcc":
                    feats = mfcc_feats
                elif f_type == "mfcc_sdc":
                    feats = sidekit.frontend.features.shifted_delta_cepstral(
                        mfcc_feats, d=d,
                        p=p, k=k)
                elif f_type == "mfcc_pca_dct":
                    feats = sidekit.frontend.features.pca_dct(
                        mfcc_feats,
                        left_ctx=left_ctx,
                        right_ctx=right_ctx,
                        p=None)
                else:
                    raise AssertionError("Feature type not found.")
            elif f_type == "ltas":
                feats = long_term_spectra(self.x[i], fs)
            elif f_type == "power_spectrum":
                feats = sidekit.frontend.features.power_spectrum(
                    self.x[i], fs=fs, win_time=win_time, shift=shift)
            else:
                raise AssertionError("Feature type not found.")

            # Normalize feat.
            if f_n is True:
                feats = feature_normalization(feats)
            # Append to the feats list.
            self.feats.append(feats)

        return self.feats, self.obj.labels

    def write_to_h5(self, feats, feats_dir):
        for i in range(len(self.obj.file_list)):
            f = h5py.File(f"{feats_dir}/{self.obj.file_list[i]}.h5", "w+")
            sidekit.frontend.io.write_hdf5(
                show=self.obj.file_list[i],
                fh=f,
                cep=feats[i],
                cep_mean=None,
                cep_std=None,
                energy=None,
                energy_mean=None,
                energy_std=None,
                fb=None,
                fb_mean=None,
                fb_std=None,
                bnf=None,
                bnf_mean=None,
                bnf_std=None,
                label=self.obj.label)

    @staticmethod
    def read_h5(obj, hdf_file_path):
        """
        Read features from h5 file. Note that file path and feature types are
        hard-coded.

        Args:
            obj: Dataset object that provide file list and labels.
            hdf_file_path: Full path of the file.

        Returns:
            A tuple that contains data and label of all file lists of object.
        """
        feats = []
        labels = obj.labels
        for i in range(len(obj.file_list)):
            filename = obj.protocol[i, 0][:9]
            file = h5py.File(hdf_file_path, "r+")
            # noinspection PyUnresolvedReferences
            feat = sidekit.frontend.io.read_dict_hdf5(hdf_file_path)
            file.close()
            feats.append(feat[f"{filename}/cep"])

        return feats, labels

    @staticmethod
    def pickle(destination_file, train=None, dev=None, eva=None):
        """
        Pickles given data into selected folder with the filename.
        """
        dump((train, dev, eva), gzip.open(destination_file, "w+"))

    @staticmethod
    def visualize(path_to_file):
        """
        Review!!!

        Reads data from data/features folder. Data type must be in pkl.gz
        format.

        Returns:
            Train, dev and eval (if exist) tuple.
        """
        try:
            with gzip.open(path_to_file, "rb") as f:
                train, dev = pickle.load(f, encoding="latin1")
                f.close()
                return train, dev
        except ValueError or EOFError:
            with gzip.open(path_to_file, "rb") as f:
                train, dev, eva = pickle.load(f, encoding="latin1")
                f.close()
                return train, dev, eva
