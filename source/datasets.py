# -*- coding: utf-8 -*-
import json
from os.path import exists, isdir, join

import numpy
import sidekit


class ASVspoof:
    """
    A class to read files and all the required information of given dataset
    (subset). Specifically, the class can be initialized to read attributes
    like protocol, file lists indices, labels etc. for the declared dataset or
    one can use static calls of the calls ßto check class usage, path etc.
    without initialization.

    See ASVspoof 2017: https://www.asvspoof.org/
    """

    def __init__(self, year=None, subset=None, dataset_path=None,
                 protocol_path=None, wav_path=None, url=""):
        """
        Initializes class for several cases(can be observed from param
        descriptions).

        Args:
            year: Specifies distribution year of dataset which can be 2015
            and 2017.
            subset: Dataset is divided three subsets which are
            train, development (dev) and evaluation(eval).
            dataset_path:
            protocol_path:
            wav_path:
            url:
        """
        with open("../config/datasets.json") as fh:
            dataset_config = json.loads(fh.read())

        self.year = year
        if self.year is None:
            self.year = 2017

        self.subset = subset
        if self.subset is None:
            self.subset = "train"

        self.url = url

        self.dataset_path = dataset_path
        if self.dataset_path is None:
            self.dataset_path = dataset_config["ASVspoof 2017"][self.subset][
                "path_to_dataset"]

        self.protocol_path = protocol_path
        if self.protocol_path is None:
            self.protocol_path = dataset_config["ASVspoof 2017"][self.subset][
                "path_to_protocol"]

        self.wav_path = wav_path
        if self.wav_path is None:
            self.wav_path = dataset_config["ASVspoof 2017"][self.subset][
                "path_to_wav"]

        if not isdir(self.dataset_path):
            assert "Directory not found.\n{}".format(self.dataset_path)

        if not exists(join(self.protocol_path)):
            assert "File not found.\n{}".format(self.protocol_path)

        if not isdir(join(self.dataset_path, self.wav_path)):
            assert "Directory not found.\n{}".format(self.wav_path)

        # Open the protocol file.
        self.protocol = []
        self.file_list = []
        self.labels = []
        self.protocol = numpy.genfromtxt(self.protocol_path, delimiter=" ",
                                         dtype=str)

        # Get file list.
        self.file_list = [item[0] for item in self.protocol]

        # Get label list.
        for item in self.protocol:
            if item[1] == "genuine":
                self.labels.append(1)
            elif item[1] == "spoof":
                self.labels.append(0)
            else:
                assert "Unknown label found."

        # Get indices of genuine and spoof file ids and lists.
        self.genuine_list = []
        self.genuineIdx = []
        self.spoof_list = []
        self.spoofIdx = []
        for index, label in enumerate(self.labels):
            if label == "genuine":
                self.genuine_list.append(1)
                self.genuineIdx.append(index)
            elif label == "spoof":
                self.spoof_list.append(0)
                self.spoofIdx.append(index)
            else:
                assert "Unknown label found."

    @staticmethod
    def load_wav(path_to_file):
        """
        Reads audio file in wav format. This does not scale signal, use
        "soundfile" module instead.

        Args:
            path_to_file:

        Returns:

        """
        x, fs, _ = sidekit.frontend.io.read_wav(path_to_file)
        return x, fs

    def load_wav_files(self):
        """
        """
        wav_files = []
        for item in self.file_list:
            absolute_path = join(self.wav_path, item)
            x, fs, _ = sidekit.frontend.io.read_wav(absolute_path)
            wav_files.append(x)

        return wav_files

    def info(self):
        """
        Prints out the dataset (subset) information.
        """
        print("{} genuine, {} spoof files (total={}) exist in {} subset of"
              "ASVSpoof dataset.\n".format(len(self.genuineIdx),
                                           len(self.spoofIdx),
                                           len(self.file_list),
                                           self.subset))
