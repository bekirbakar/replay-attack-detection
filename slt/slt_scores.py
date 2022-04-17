# -*- coding: utf-8 -*-
import math

import numpy
import numpy as np
import sidekit


def mean_of_frames(probabilities, indexes):
    """
    Finds means of frames for the frame wise classification.

    :param probabilities: one dimensional array.
    :param indexes: mean indexes.
    :return: means.
    """
    means = []
    index1 = 0
    for i in range(0, len(indexes)):
        index0 = index1
        index1 = indexes[i] + index0
        means.append(numpy.mean(probabilities[index0:index1]))

    return means


def scoring(score_file=None, probabilities=None, labels=None, file_list=None,
            indexes=None):
    """
    Provides score operations.

    Args:
        score_file: Score file to get tar, non-tar or eer.
        probabilities: Predictions in categorical form.
        labels: Can be spoof (0) or genuine (1).
        file_list: Name of files.
        indexes: Feature indexes for frame wise classification.

    Returns:
        Target and non-target scores, eer and score file.
    """
    # Extract target and non-target scores.
    tar = []
    nontar = []
    if score_file is not None:
        for i in range(0, len(score_file)):
            if score_file[(i, 1)] == "genuine" or score_file[(i, 1)] == "1":
                tar.append(score_file[(i, 2)])
            elif score_file[(i, 1)] == "spoof" or score_file[(i, 1)] == "0":
                nontar.append(score_file[(i, 2)])
            else:
                raise AssertionError(
                    "Some of the values are missing in score file.")
    else:
        if labels is None:
            raise AssertionError("Labels must be specified.")
        if probabilities is None:
            raise AssertionError("Probabilities must be specified.")

        # Subtracts second columns from first columns.
        # p_diff = []
        p_diff = np.zeros([len(probabilities)], dtype=float)
        for i in range(len(probabilities)):
            # p_diff.append(probabilities[i][0] - probabilities[i][1])
            try:
                p_diff[i] = math.log(probabilities[i][1]) - math.log(
                    probabilities[i][0])
            except ValueError:
                eps = 2.2204e-16
                p_diff[i] = math.log(probabilities[i][1] + eps) - math.log(
                    probabilities[i][0] + eps)
                # Frame wise classification.
        if indexes is not None:
            p_diff = mean_of_frames(p_diff, indexes)
        else:
            pass
        # Divide scores as nan and non-tar.
        for i in range(0, len(labels)):
            if labels[i] == 1:
                tar.append(str(p_diff[i]))
            elif labels[i] == 0:
                nontar.append(str(p_diff[i]))
            else:
                raise AssertionError("Some of the labels are missing.")
        # Creates score file(filename-label-score).
        score_file = []
        for i in range(0, len(p_diff)):
            if file_list is not None:
                if len(p_diff) != len(labels):
                    raise AssertionError(
                        "Labels and scores are not in same length.")
                score_file.append(
                    str(file_list[i]) + " " + str(labels[i]) + " " + str(
                        p_diff[i]))
            else:
                score_file.append(
                    "None" + " " + str(labels[i]) + " " + str(p_diff[i]))
    # Calculates equal error rate.
    pmiss, pfa = sidekit.bosaris.detplot.rocch(np.asarray(tar),
                                               np.asarray(nontar))
    eer = sidekit.bosaris.detplot.rocch2eer(pmiss, pfa) * 100
    return round(eer, 2), tar, nontar, score_file


def save(file_path=None, file_name=None, data_to_save=None):
    if file_path is not None:
        file_path = file_path + "/" + file_name
    else:
        file_path = file_name

    f = open(file_path, "w+")

    for i in range(0, len(data_to_save)):
        data_to_save[i] = data_to_save[i] + "\n"
        f.write(str(data_to_save[i]))
    f.close()


def read(filename):
    """
    Reads score file.

    Args:
        filename: Score filename in text format.

    Returns:
        ndarray object of numpy module as score file.
    """
    score_file_path = "data/scores/" + filename
    score_file = numpy.genfromtxt(score_file_path, delimiter=" ", dtype=str)
    return score_file
