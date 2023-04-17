# -*- coding: utf-8 -*-
import json
import os
import pickle
from math import ceil

import joblib
import keras
import numpy as np
import pandas
import scipy.io
import sidekit
import sio
import soundfile as sf
from keras.backend import clear_session
from obspy.signal.util import enframe, next_pow_2
from python_speech_features.sigproc import preemphasis
from scipy import signal

from source.asvspoof_data import ASVspoof
from source.data_io_utils import flush_process_info, clear_dir

with open("./config/datasets.json") as fh:
    dataset_config = json.loads(fh.read())


def mvn(feat):
    if len(feat.shape) == 1:
        norm_feat = feat.copy()

        # Find mean and variance.
        mu = np.mean(feat)
        sigma = np.std(feat)

        # Subtract mu from each element.
        norm_feat -= mu

        # Divide by sigma.
        norm_feat /= sigma
    else:
        norm_feat = np.zeros([feat.shape[0], feat.shape[1]], dtype=float)
        for i in range(feat.shape[1]):
            mu = np.mean(feat[:, i])
            sigma = np.std(feat[:, i])
            norm_feat[:, i] = (feat[:, i] - mu) / sigma

    return norm_feat


def ltas(x, nfft, fs=16000, fc=None):
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Pre-emphasis on signal.
    x = preemphasis(x, 0.97)

    # Calculate frame and shift length in terms of samples.
    frame_len = int((20 * fs) / 1000)
    frame_step = frame_len // 2

    # Apply hamming window and split signal into frames.
    frames, _, _ = enframe(x + eps, np.hamming(frame_len), frame_step)

    # Crop low frequencies.
    start = int((fc * nfft) / fs) if fc is not None else 1
    end = int(nfft / 2)
    length = int(end - start)

    # A place holder for ltas.
    spectrum = np.zeros([len(frames), length], dtype=float)

    # N point FFT of signal.
    for frame in range(len(frames)):
        fft = np.fft.fft(frames[frame], nfft)
        fft_shift = fft[1:int(nfft / 2) + 1]

        # Fourier magnitude spectrum computation.
        log_magnitude = np.log(abs(fft_shift) + eps)
        spectrum[frame] = log_magnitude[start:end]

    # Concatenate mean and variance statistics.
    mu = np.mean(spectrum, axis=0)
    sigma = np.std(spectrum, axis=0)
    return np.concatenate((mu, sigma), axis=0)


# noinspection DuplicatedCode
def split_frames(feats, labels):
    # Get info, length of every mXn dimensional list element.
    count = 0
    info = []
    for i in range(len(feats)):
        info.append(feats[i].shape[0])
        count += info[i]

    # Calculate index for each list and store according to index into x and y.
    x = []  # Declare x to store features.
    y = []  # Declare y to store labels.
    index = 0
    j = 0
    for i in range(len(feats)):
        # Store data into buffer.
        index += j
        buffer = feats[i]

        # Append data into x and labels into y.
        for j in range(len(buffer)):
            x.append(buffer[j])
            y.append(labels[i])
        j += 1
        flush_process_info("Completed.", i, len(feats))

    return x, y, info


def reverse_arr(arr):
    x = arr.shape[1]
    y = arr.shape[0]
    ret_arr = np.zeros([x, y], dtype=float)
    for i in range(x):
        ret_arr[i] = arr[:, i]

    return ret_arr


def read_matlab_file(filename):
    # mat_contents = sio.loadmat(filename + ".mat")["features2"]
    # noinspection PyUnresolvedReferences
    mat_contents = sio.loadmat(filename + ".mat")["features2"]
    mat_contents = mvn(mat_contents)

    return reverse_arr(mat_contents)


def ltss(subset_list, normalize=False):
    for subset in subset_list:
        dataset = ASVspoof(
            2017,
            subset,
            dataset_config["ASVspoof 2017"][subset]["path_to_dataset"],
            dataset_config["ASVspoof 2017"][subset]["path_to_protocol"],
            dataset_config["ASVspoof 2017"][subset]["path_to_wav"],
        )
        file_list = dataset.file_list
        path_to_wav = dataset_config["ASVspoof 2017"][subset]["path_to_wav"]
        labels = dataset.labels

        # Extract features.
        feats = []
        for file in range(len(file_list)):
            x, fs = sf.read(path_to_wav + file_list[file] + ".wav")
            feat = ltas(x, nfft=1024, fc=4000)

            # Normalize.
            if normalize is True:
                feat = mvn(feat)

            # Append data into the list.
            feats.append(feat)
            flush_process_info("Completed.", file, len(file_list))
        data = feats, labels

        # Change directory.
        wor_dir = os.getcwd()
        os.chdir("../data/features/ltas/")

        # Save data to disk.
        f = open("ltas_" + subset, "wb+")
        pickle.dump(data, f)

        # Change back to working directory.
        os.chdir(wor_dir)


def cqcc(norm=False):
    subset_list = ["eval"]
    for subset in subset_list:
        path = "../../data/features/cqcc/" + subset

        dataset = ASVspoof(
            2017,
            subset,
            dataset_config["ASVspoof 2017"][subset]["path_to_dataset"],
            dataset_config["ASVspoof 2017"][subset]["path_to_protocol"],
            dataset_config["ASVspoof 2017"][subset]["path_to_wav"],
        )
        labels = dataset.labels
        wor_dir = os.getcwd()
        os.chdir(path)
        cqcc_feats = []
        for file in dataset.file_list:
            feat = read_matlab_file(file)
            if norm is True:
                norm_feat = mvn(feat)
                cqcc_feats.append(norm_feat)
            else:
                cqcc_feats.append(feat)

        # Split frames.
        cqcc_feats = split_frames(cqcc_feats.copy(), labels)

        # Save to disk.
        os.chdir("../source")
        file = open("cqcc_" + subset, "wb+")
        pickle.dump(cqcc_feats, file, protocol=3)
        os.chdir(wor_dir)

        del cqcc_feats, labels, path, wor_dir


def mfcc(norm=False):
    subset_list = ["eval"]
    for subset in subset_list:
        path = "../../data/features/mfcc/" + subset
        dataset = ASVspoof(
            2017,
            subset,
            dataset_config["ASVspoof 2017"][subset]["path_to_dataset"],
            dataset_config["ASVspoof 2017"][subset]["path_to_protocol"],
            dataset_config["ASVspoof 2017"][subset]["path_to_wav"],
        )
        labels = dataset.labels
        wor_dir = os.getcwd()
        os.chdir(path)
        mfcc_feats = []

        for file in dataset.file_list:
            feat = read_matlab_file(file)
            if norm is True:
                norm_feat = mvn(feat)
                mfcc_feats.append(norm_feat)
            else:
                mfcc_feats.append(feat)

        # Split frames.
        mfcc_feats = split_frames(mfcc_feats.copy(), labels)

        # Save to disk.
        os.chdir("../source/")
        file = open("mfcc_" + subset, "wb+")
        pickle.dump(mfcc_feats, file)
        os.chdir(wor_dir)

        del mfcc_feats, labels, path, wor_dir


def high_pass_filter(x, order=10, f_cut=3000):
    # Design a high-pass filter.
    f_cut = f_cut / 8000
    [b, a] = signal.butter(order, f_cut, "high")

    # Apply filter to signal.
    x = signal.lfilter(b, a, x)

    return x


# noinspection DuplicatedCode
def split_feats(feats, labels):
    # noinspection DuplicatedCode
    """
    Splits given ceps into an array list. One element of feats lists is stored
    in mxn numpy.ndarray (n is ceps number), the function store this data into
    nx1 numpy.ndarray.

    Args:
        feats: Feature list, every list element must contain numpy.ndarray.
        labels: Corresponding label of feature.

    Returns:
        A tuple for features, labels and dimension info of each list elements.
        """
    # Get info, length of every mXn dimensional list element.
    count = 0
    info = []
    for i in range(len(feats)):
        info.append(feats[i].shape[0])
        count += info[i]

    # Calculate index for each list and store according to index into x and y.
    x = []  # Declare x to store features.
    y = []  # Declare y to store labels.
    index = 0
    j = 0
    for i in range(len(feats)):
        # Store data into buffer.
        index += j
        buffer = feats[i]

        # Append data into x and labels into y.
        for j in range(len(buffer)):
            x.append(buffer[j])
            y.append(labels[i])
        j += 1
        flush_process_info("Completed.", i, len(feats))

    return x, y, info


def matrix_to_vector(matrix):
    """
    Splits given matrix (m,n) into vector ((mxn),1).

    Args:
        matrix: Matrix shaped input data.

    Returns:
        Data vector.
    """
    # Find the shape of matrix.
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]

    # Vector size to be created.
    v_size = row_size * col_size
    vector = np.ones([v_size], dtype=float)
    for i in range(len(matrix)):
        a = i * col_size
        b = (i + 1) * col_size
        vector[a:b] = matrix[i, :]

    return vector


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
        feat_lengths = np.zeros([len(feats)], dtype=int)
        for i in range(len(feats)):
            feat_lengths[i] = len(feats[i])
        # noinspection PyArgumentList
        min_len = feat_lengths.min()

        # Crop every row that has longer than minimum length.
        for i in range(len(feats)):
            fixed_feats.append(matrix_to_vector(feats[i]))
            fixed_feats[i] = fixed_feats[i][:min_len]
            flush_process_info("Completed.", i, len(feats))
    elif method == "2":
        feat_lengths = [len(feats[i]) for i in range(len(feats))]
        mean_of_len = ceil(sum(feat_lengths) / len(feats))

        # Find ceps number.
        ceps_number = len(feats[0][0])

        # Scale every row to mean value of length by cropping longer and
        # adding values(from first values) to smaller ones.
        for i in range(len(feats)):
            buffer = np.zeros([mean_of_len, ceps_number], dtype=float)

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
            elif missing > 0:
                # Fill with first values.
                for j in range(ceps_number):
                    buffer[j, :] = feat[j, 0:mean_of_len]
            fixed_feats.append(buffer)
            flush_process_info("Completed.", i, len(feats))
    elif method == "3":
        # Convert matrix to vector.
        for i in range(len(feats)):
            buffer = feats[i]
            buffer = matrix_to_vector(buffer)
            fixed_feats.append(buffer)
            flush_process_info("Completed.", i, len(feats))
    else:
        print("Option does not exist. \n")

    return fixed_feats


def feature_normalization(feat):
    if len(feat.shape) == 1:
        mu = np.mean(feat)
        sigma = np.std(feat)
        normalized_feat = np.zeros([len(feat)], dtype=float)
        for i in range(len(feat)):
            normalized_feat[i] = (feat[i] - mu) / sigma
    else:
        normalized_feat = np.zeros([feat.shape[0], feat.shape[1]], dtype=float)
        for i in range(feat.shape[1]):
            mu = np.mean(feat[:, i])
            sigma = np.std(feat[:, i])
            for j in range(feat.shape[0]):
                normalized_feat[j][i] = (feat[j][i] - mu) / sigma

    return normalized_feat


# noinspection DuplicatedCode
def long_term_spectra(x, fs):
    """
    Extracts long-term spectral statistics of given signal. For more details on
    implementation this method: See Section III of "Presentation Attack
    Detection Using Long-Term Spectral Statistics for Trustworthy Speaker
    Verification"

    Args:
        x: Pure speech signal (note that pre-emphasis, windowing etc. applied
        to signal inside the function).
        fs: Sample rate of the signal.

    Returns:
        Long-term spectral statistics of x.
        """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Pre-emphasis on signal.
    x = preemphasis(x, 0.97)  # x = sidekit.frontend.vad.pre_emphasis(x, 0.97)

    # Calculate frame and shift length.
    frame_len = 20 * fs / 1000
    frame_step = frame_len / 2

    # Apply hamming window and split signal into frames.
    frames, _, no_win = enframe(x + eps, np.hamming(frame_len),
                                int(frame_step))
    # Number of fft points.
    nfft = next_pow_2(frame_len)

    # A place holder for ltas.
    ltas_ = np.zeros([no_win, int(nfft / 2) + 1], dtype=float)

    # N point fft of signal.
    for i in range(no_win):
        y = frames[i, :]
        abs_y = abs(np.fft.fft(y, nfft))
        ltas_[i, :] = abs_y[1:int(nfft / 2) + 2]

    # Declare mu, sigma and feats.
    mu = np.zeros([int(nfft / 2)], dtype=float)
    sigma = np.zeros([int(nfft / 2)], dtype=float)
    feat = np.zeros([nfft], dtype=float)

    # Feats: Mean and standard deviation of ltas.
    for i in range(int(nfft / 2)):
        ltas_log = np.log([ltas_[:, i] + eps])
        mu[i] = np.mean(ltas_log)
        sigma[i] = np.std(ltas_log)
        feat[i + int(nfft / 2)] = sigma[i]
        feat[i] = mu[i]

    return feat


def get_frames(x, fs):
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Pre-emphasis on signal.
    x = preemphasis(x, 0.97)  # x = sidekit.frontend.vad.pre_emphasis(x, 0.97)

    # Calculate frame and shift length.
    frame_len = 20 * fs / 1000
    frame_step = frame_len / 2

    # Apply hamming window and split signal into frames.
    frames, _, _ = enframe(x + eps, np.hamming(frame_len), int(frame_step))

    return frames


def deep_features(feature_list, dataset_index, subset_list):
    start = 0
    end = 0
    indexes = 0
    for feature in feature_list:
        excel_filename = feature + ".xlsx"

        # Read parameters from excel file(.xlsx).
        xl = pandas.ExcelFile("../data/EXCELS/" + excel_filename)
        df = xl.parse("mlp_parameters")
        dataset_name = df["dataset"][dataset_index]
        path = "/" + dataset_name.upper() + "/" + dataset_name

        # Load model.
        filename = "../data/MODELS/best_model_" + dataset_name + str(
            dataset_index) + ".h5"
        model = keras.models.load_model(filename)

        # model.summary() # print model summary.
        # Get last hidden from deep NN.
        intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                      outputs=model.get_layer(
                                                          "hidden3").output)
        for subset in subset_list:
            dataset = ASVspoof(
                2017,
                subset,
                dataset_config["ASVspoof 2017"][subset]["path_to_dataset"],
                dataset_config["ASVspoof 2017"][subset]["path_to_protocol"],
                dataset_config["ASVspoof 2017"][subset]["path_to_wav"],
            )
            file_list = dataset.file_list
            mat_file = "../../data/features/" + \
                       dataset_name.upper() + "/mat_files/" + subset + "/"
            # Load data.
            if feature == "ltas":
                data = joblib.load("../../data/features/" + path + "_" +
                                   subset)[0]
            else:
                data = joblib.load("../../data/features/" + path + "_" +
                                   subset)
                data, indexes = data[0], data[2]
                start = 0
                end = indexes[0]
            # Convert to numpy array.
            data = np.asarray(data)
            # Convert data types to float32.
            data = data.astype("float32")
            # Loop through files.
            for file in range(len(file_list)):
                # Extract features from intermediate layer.
                buffer = data[file:(file + 1)] if feature == "ltas" else data[start:end]
                intermediate_output = intermediate_layer_model.predict(buffer)
                # Save feature per file.
                mdict = {"data": intermediate_output}
                scipy.io.savemat(mat_file + file_list[file] + ".mat", mdict,
                                 appendmat=True, format="5",
                                 long_field_names=False, do_compression=False,
                                 oned_as="row")
                if feature != "ltas":
                    start = end
                    end = indexes[file] + start
        # Delete model to release gpu memory.
        clear_session()


def load_deep_feat(dataset, subset_list):
    # Load mat file back.
    mat_file = "../../data/features/" + dataset.upper() + "/mat_files/" + \
               subset_list[0] + "/"
    file_list = ASVspoof(
        2017,
        subset_list[0],
        dataset_config["ASVspoof 2017"][subset_list[0]]["path_to_dataset"],
        dataset_config["ASVspoof 2017"][subset_list[0]]["path_to_protocol"],
        dataset_config["ASVspoof 2017"][subset_list[0]]["path_to_wav"],
    ).file_list
    return scipy.io.loadmat(mat_file + file_list[1000])["data"]


def extract_features(feature_list, subset_list):
    # Select params for feature types.
    # Feature_type, normalization, filter, context, frame_wise.
    for i in range(len(feature_list)):
        f_type = feature_list[i]
        path = "../../data/features/" + f_type.upper() + "/"
        file = path + f_type
        # Select params for feature types.
        # normalization, filter, context, frame_wise.
        if f_type == "ltas":
            params = [True, False, False, False]
        elif f_type in ["mfcc", "power_spectrum"]:
            params = [True, False, False, True]
        elif f_type == "frames":
            params = [False, False, False, True]
        else:
            raise AssertionError("Feature type not found.")
        # Loop through feature list.
        for j in range(len(subset_list)):
            s = subset_list[i]
            subset = ASVspoof(
                2017, subset_list[j],
                dataset_config["ASVspoof 2017"][s]["path_to_dataset"],
                dataset_config["ASVspoof 2017"][s]["path_to_protocol"],
                dataset_config["ASVspoof 2017"][s]["path_to_wav"])

            feats, labels = Feature(subset).extract(f_type,
                                                    f_n=params[0],
                                                    apply_filter=params[1],
                                                    context=params[2])
            data = split_feats(feats, labels) if params[3] is True else (feats, labels)
            joblib.dump(data, file + "_" + subset_list[j])
            del data, feats, labels


class Feature(object):
    """
    A class to extract features from speech signal using SIDEKIT package and
    some other functions created above. In addition, this class provide to
    read, write, pickle, unpickle, visualize and other utilities for
    manipulating extracted features.
    """

    def __init__(self, obj):
        self.obj = obj
        self.x = []
        self.feats = []
        # Sidekit extractor.
        self.extractor = sidekit.FeaturesExtractor(
            audio_filename_structure=self.obj.path_to_wav + "{}.wav",
            feature_filename_structure="../../data/features/sidekit/" +
                                       self.obj.subset + "/" + "{}.h5",
            sampling_frequency=16000,
            lower_frequency=6000,
            higher_frequency=8000,
            filter_bank="log",
            filter_bank_size=27,
            window_size=0.025,
            shift=0.01,
            ceps_number=19,
            vad="snr",
            snr=40,
            pre_emphasis=0.97,
            save_param=["cep", "energy"],
            keep_all_features=True)
        # Sidekit server.
        self.server = sidekit.FeaturesServer(
            features_extractor=None,
            feature_filename_structure="../../data/features/SIDEKIT/" +
                                       obj.subset + "/{}.h5",
            # feature_filename_structure=None,
            sources=None,
            dataset_list=["cep", "energy"],
            mask=None,
            feat_norm=None,
            global_cmvn=False,
            dct_pca=False,
            dct_pca_config=None,
            sdc=False,
            sdc_config=None,
            delta=True,
            double_delta=True,
            delta_filter=None,
            context=(5, 5),
            traps_dct_nb=None,
            rasta=False,
            keep_all_features=False)

    def sidekit_extractor(self):
        """
        Extracts mfcc and saves to .h5 file according to parameters below.
        This method adapted from SIDEKIT tutorials.

        See
        https://www-lium.univ-lemans.fr/sidekit/tutorial/featuresextractor.html
        """
        # Delete old files in directory.
        clear_dir("../../data/features/SIDEKIT/" + self.obj.subset + "/",
                  ".h5")
        # Extract and save features.
        for i in range(len(self.obj.file_list)):
            self.extractor.save(self.obj.file_list[i])
            flush_process_info("Completed.", i, len(self.obj.file_list))

    def extract(self, f_type, fs=16000, f_n=False, apply_filter=False,
                context=False):
        """
        Extracts features.

        Args:
            f_type: Can be "mfcc", "mfcc_sdc", "mfcc_pca_sdc", "ltas",
            "power_spectrum".
            fs: Sampling frequency.
            f_n: Feature normalization.
            apply_filter: Apply high pass filter to signal.
            context: Context number (example (5,5)).

        Returns:
            Features and labels.
        """
        # Extract selected method to data in x.
        for i in range(len(self.obj.file_list)):
            x = sf.read(self.obj.path_to_wav + self.obj.file_list[i] + ".wav")[
                0]
            # print(str(self.obj.path_to_wav) + str(self.obj.file_list[i]) +
            # ".wav")
            # Add random noise on elements to avoid dividing values to zero.
            np.random.seed(0)
            rand_noise = 0.0001 * np.random.randn(1)
            x = x + rand_noise
            # Apply high pass filter on signal.
            if apply_filter is True:
                x = high_pass_filter(x)
            # Feature extraction.
            if f_type == "mfcc":
                mfcc_ = sidekit.frontend.features.mfcc(x,
                                                       lowfreq=6000,
                                                       maxfreq=8000,
                                                       nlinfilt=27,
                                                       nlogfilt=0,
                                                       nwin=0.025,
                                                       fs=fs,
                                                       nceps=19,
                                                       shift=0.01,
                                                       get_spec=False,
                                                       get_mspec=False,
                                                       prefac=0.97)
                # Add log-energy as first coefficient.
                mfcc_ = np.insert(mfcc_[0], 0, values=mfcc_[1], axis=1)
                feat = self.server.get_context(mfcc_)[0] if context is True else mfcc_
            elif f_type == "ltas":
                feat = long_term_spectra(x, fs)
            elif f_type == "power_spectrum":
                feat = sidekit.frontend.features.power_spectrum(x, fs=fs)
                # Add log-energy as first coefficient.
                feat = np.insert(feat[0], 0, values=feat[1], axis=1)
            elif f_type == "frames":
                feat = get_frames(x, fs)
            else:
                raise AssertionError("Feature type not found.")
            if f_n is True:
                feat = feature_normalization(feat)
            # Append to the feats list.
            self.feats.append(feat)
            flush_process_info("Completed.", i, len(self.obj.file_list))
        return self.feats, self.obj.labels


if __name__ == "__main__":
    extract_features(feature_list=["ltas"],
                     subset_list=["train", "dev", "eval"])
    extract_features(feature_list=["mfcc"],
                     subset_list=["train", "dev", "eval"])
