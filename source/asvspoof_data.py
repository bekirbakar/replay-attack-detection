"""
This module provides functionality for loading and processing the ASVspoof
dataset. The main class, ASVspoof, reads files and the required information of a
given dataset (subset). Functions are provided to load and share the dataset,
read data from specified datasets and subsets, and read WAV files. 

See ASVspoof2017: https://www.asvspoof.org/
"""

import gzip
import json
import pickle
from os.path import exists, isdir, join
from typing import List, Tuple

import keras.utils
import numpy
import numpy as np
import sidekit
import theano
import theano.tensor as t


def load(dataset: str) -> tuple:
    """
    Loads the dataset and returns shared variables for train and dev sets.

    Args:
        dataset: The path of the dataset file.

    Returns:
        rval: A tuple containing shared variables for the train and dev sets.
    """
    try:
        with gzip.open(dataset, 'rb') as f:
            train, dev, _ = pickle.load(f, encoding='latin1')
    except (ValueError, EOFError):
        with gzip.open(dataset, 'rb') as f:
            train, dev = pickle.load(f, encoding='latin1')

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, t.cast(shared_y, 'int32')

    train_x, train_y = shared_dataset(train)
    dev_x, dev_y = shared_dataset(dev)

    rval = [(train_x, train_y), (dev_x, dev_y)]

    return rval


def read_data(dataset: str, subset: str) -> dict:
    """
    Reads the data from the specified dataset and subset.

    Args:
        dataset: The name of the dataset.
        subset: The subset of the dataset.

    Returns:
        A dictionary containing data, indexes, labels, and other information
        from the specified dataset and subset.
    """
    # Read data.
    path = f"feats/{dataset.upper()}/"
    with open(path + dataset + "_" + subset, "rb") as file:
        data = pickle.load(file)

    # Convert to numpy array.
    x_data = np.asarray(data[0]).astype("float32")
    y_data = np.asarray(data[1]).astype("float32")
    y_data = keras.utils.to_categorical(y_data, 2)

    # Get labels and file_list.
    with open("./config/datasets.json") as fh:
        d = json.loads(fh.read())

    dataset = ASVspoof(2017,
                       subset,
                       d["ASVspoof2017"][subset]["path_to_dataset"],
                       d["ASVspoof2017"][subset]["path_to_protocol"],
                       d["ASVspoof2017"][subset]["path_to_wav"])

    file_list = dataset.file_list
    labels = dataset.file_list

    # Frame-wise classification.
    if dataset in ["mfcc", "cqcc"]:
        indexes = data[2]
    elif dataset == "ltas":
        indexes = None
    else:
        raise AssertionError("Dataset not found.")

    mini_batch_size = len(
        x_data) // 50 if len(x_data) >= 10000 else len(x_data)
    del data

    # Store values in data dictionary.
    return {
        "dataset": dataset,
        "indexes": indexes,
        "x_data": x_data,
        "y_data": y_data,
        "labels": labels,
        "file_list": file_list,
        "mini_batch_size": mini_batch_size
    }


class ASVspoof:
    """
    A class to read files and all the required information of a given dataset
    (subset). Specifically, the class can be initialized to read attributes
    like protocol, file lists indices, labels etc. for the declared dataset or
    one can use static calls of the class to check class usage, path etc.
    without initialization.

    See ASVspoof2017: https://www.asvspoof.org/
    """

    def __init__(self, year: int = None, subset: int = None,
                 dataset_path: str = None, protocol_path: str = None,
                 wav_path: str = None, url: str = "") -> None:
        """
        Initializes class for several cases (can be observed from param
        descriptions).

        Args:
            year: Specifies distribution year of dataset which can be 2015
            and 2017.
            subset: Dataset is divided into three subsets which are
            train, development (dev) and evaluation (eval).
            dataset_path:
            protocol_path:
            wav_path:
            url:
        """
        with open("../config/datasets.json") as fh:
            dataset_config = json.loads(fh.read())

        self.year = year if year is not None else 2017
        self.subset = subset if subset is not None else "train"
        self.url = url

        self.dataset_path = dataset_path if dataset_path is not None\
            else dataset_config["ASVspoof2017"][self.subset]["path_to_dataset"]
        self.protocol_path = protocol_path if protocol_path is not None\
            else dataset_config["ASVspoof2017"][self.subset]["path_to_protocol"]
        self.wav_path = wav_path if wav_path is not None else dataset_config[
            "ASVspoof2017"][self.subset]["path_to_wav"]

        if not isdir(self.dataset_path):
            raise AssertionError(f"Directory not found.\n{self.dataset_path}")

        if not exists(join(self.protocol_path)):
            raise AssertionError(f"File not found.\n{self.protocol_path}")

        if not isdir(join(self.dataset_path, self.wav_path)):
            raise AssertionError(f"Directory not found.\n{self.wav_path}")

        # Open the protocol file.
        self.protocol = numpy.genfromtxt(
            self.protocol_path, delimiter=" ", dtype=str)

        # Get file list and label list.
        self.file_list = [item[0] for item in self.protocol]
        self.labels = [1 if item[1] ==
                       "genuine" else 0 for item in self.protocol]

        # Get indices of genuine and spoof file ids and lists.
        self.genuineIdx = [idx for idx,
                           label in enumerate(self.labels) if label == 1]
        self.spoofIdx = [idx for idx, label in enumerate(
            self.labels) if label == 0]
        self.genuine_list = [1 for _ in self.genuineIdx]
        self.spoof_list = [0 for _ in self.spoofIdx]

    @staticmethod
    def load_wav(path_to_file: str) -> Tuple[numpy.ndarray, int]:
        """
        Reads an audio file in WAV format and returns the audio signal and
        sample rate.

        Args:
            path_to_file (str): The path to the audio file.

        Returns:
            x (numpy.ndarray): The audio signal.
            fs (int): The sample rate of the audio file.
        """
        x, fs, _ = sidekit.frontend.io.read_wav(path_to_file)
        return x, fs

    def load_wav_files(self) -> List[numpy.ndarray]:
        """
        Reads all WAV files from the file list associated with the dataset and
        returns a list of audio signals.

        Returns:
            wav_files (list): A list of audio signals (numpy.ndarray)
            corresponding to the files in the dataset.
        """
        wav_files = []
        for item in self.file_list:
            absolute_path = join(self.wav_path, item)
            x, _, _ = sidekit.frontend.io.read_wav(absolute_path)
            wav_files.append(x)
        return wav_files
