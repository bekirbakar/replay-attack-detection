# -*- coding: utf-8 -*-
import math
import random

import numpy
import numpy as np
# noinspection PyPackageRequirements
from external.detplot import DetPlot
from sidekit.bosaris.detplot import rocch, rocch2eer


def save(data, path_to_file, filename=random.choice("abcdefgh")):
    if type(data) != list:
        assertion_text = "\n" + "Data type must be list!"
        raise AssertionError(assertion_text)

    f = open(path_to_file + filename + ".txt", "w+")
    for i in range(0, len(data)):
        f.write(str(data[i]) + "\n")

    f.close()


def load(file_path):
    return numpy.genfromtxt(file_path, delimiter=" ", dtype=str)


def create(values, labels, file_list=None, indexes=None):
    """
    Combines given parameters to create a score file.

    Args:
        values: Predictions in categorical form.
        labels: Spoof (0) or genuine (1) labels to divide scores into two
        categories as target and non-target scores.
        file_list: Name of files to put first column of score file.
        indexes: Feature indexes for frame wise classification.

    Returns:
        Score file in --filename-label-score-- format.
    """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Calculate log-likelihood ratio.
    llr_values = []
    for i in range(len(values)):
        try:
            a = math.log(values[i][1])
            b = math.log(values[i][0])
        except ValueError:
            a = math.log(values[i][1] + eps)
            b = math.log(values[i][0] + eps)
        llr_values.append(a - b)

    # Frame-wise classification.
    if indexes is not None:
        llr_values = mean_of_frames(llr_values, indexes)

    # Divide scores into tar and non-tar categories.
    tar = []
    nontar = []
    for i in range(0, len(labels)):
        if labels[i] == 1 or labels[i] == "genuine":
            tar.append(str(llr_values[i]))
        elif labels[i] == 0 or labels[i] == "spoof":
            nontar.append(str(llr_values[i]))
        else:
            raise AssertionError("\n" + "Some of the labels are missing.")

    # Creates score file(filename-label-score) and check if there are missing
    # values.
    if len(llr_values) != len(labels):
        assertion_text = "\nLabels and scores are not in same length."
        raise AssertionError(assertion_text)

    # Create score file.
    score_file = []
    for i in range(0, len(llr_values)):
        labels_as_string = ""
        if str(labels[i]) == "1":
            labels_as_string = "genuine "
        elif str(labels[i]) == "0":
            labels_as_string = "spoof "
        else:
            labels_as_string = (labels_as_string[i])
        str_llr = str(llr_values[i])
        if file_list is not None:
            str_file_list = str(file_list[i]) + " "
        else:
            str_file_list = "None"
        score_file.append(str_file_list + labels_as_string + str_llr)

    return score_file


def calculate_eer(data, from_file=False, tar=None, nontar=None):
    """
    Calculates equal error rate from given file of predictions stored.

    Args:
        data: Score file in --filename-label-score-- format.
        from_file: Boolean indicator whether score file or data fed.
        nontar: Non target scores.
        tar: Target scores.

    Returns:
        Rounded (2) equal error rate.
    """

    if from_file is True:
        data = load(data)

    if tar is None and nontar is None:
        if type(data) is list:
            data = np.genfromtxt(data, delimiter=" ",
                                 dtype=str)

        # Extract target and non-target scores.
        tar = []
        nontar = []
        for line in range(0, len(data)):
            if data[(line, 1)] == "genuine" or \
                    data[(line, 1)] == "1":
                tar.append(data[(line, 2)])
            elif data[(line, 1)] == "spoof" or \
                    data[(line, 1)] == "0":
                nontar.append(data[(line, 2)])
            else:
                assertion_text = "\n" + "Some of the values are missing in" \
                                        "score file."
                raise AssertionError(assertion_text)

    # Calculate equal error rate (eer).
    pmiss, pfa = rocch(np.asarray(tar), np.asarray(nontar))
    eer = rocch2eer(pmiss, pfa) * 100

    return round(eer, 2)


def mean_of_frames(values, indexes):
    """
    Finds mean of frames (for frame wise classification).

    Args:
        values: One dimensional array.
        indexes: Mean indexes.

    Returns:
        Means
    """
    means = []
    end = 0
    for i in range(0, len(indexes)):
        start = end
        end = start + indexes[i]
        means.append(numpy.mean(values[start:end]))

    return means


def rocch2eer_custom(pmiss, pfa):
    """
    Simple edit of sidekit rocch2eer function. See external.py in detplot older
    for explanation.

    Args:
        pmiss: The vector of miss probabilities.
        pfa: The vector of false-alarm probabilities.

    Returns:
        The Equal Error Rate
    """
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            "Pmiss and pfa have to be sorted."

        # noinspection DuplicatedCode
        xy = numpy.column_stack((xx, yy))
        dd = numpy.dot(numpy.array([1, -1]), xy)
        if numpy.min(numpy.abs(dd)) == 0:
            eer_seg = 0
        else:
            seg = numpy.linalg.solve(xy, numpy.array([[1], [1]]))
            eer_seg = 1 / (numpy.sum(seg))

        eer = max([eer, eer_seg])

    return eer


def tar_nontar(score_file):
    """
    Extract nontar and tar values from score file.

    Args:
        score_file: Path to the score file.

    Returns:
        Nontar and tar scores.
    """
    tar = []
    nontar = []
    for i in range(0, len(score_file)):
        if score_file[(i, 1)] == "genuine" or score_file[(i, 1)] == "1":
            tar.append(score_file[(i, 2)])
        elif score_file[(i, 1)] == "spoof" or score_file[(i, 1)] == "0":
            nontar.append(score_file[(i, 2)])
        else:
            raise AssertionError("Some of the values are missing in score"
                                 "file.")

    # Return tar and non target scores.
    return tar, nontar


def det_plot(tar, non_tar, sys_name=""):
    # Convert to numpy arrays.
    tar = numpy.asarray(tar)
    non_tar = numpy.asarray(non_tar)

    # Plot the DET curve
    # prior = sidekit.logit_effective_prior(0.01, 10, 1)
    dp = DetPlot(plot_title="")
    dp.set_system(numpy.asarray(tar), numpy.asarray(non_tar),
                  sys_name=sys_name)
    dp.create_figure()
    dp.plot_rocch_det(0)
    # dp.plot_DR30_miss()
    # dp.plot_DR30_both(idx=0)
    # dp.plot_mindcf_point(prior, idx=0)

    param = []
    for i in range(0, len(param)):
        dp.set_system(numpy.asarray(tar), numpy.asarray(non_tar),
                      sys_name=sys_name)


def subtraction(probabilities):
    result = []
    for i in range(len(probabilities)):
        result.append(probabilities[i][0] - probabilities[i][1])

    return result


def fix_scores(probabilities):
    fixed_p = []
    labels = []
    for i in range(len(probabilities)):
        buffer = probabilities[i][0]
        buffer1 = probabilities[i][1]
        for j in range(0, len(buffer)):
            fixed_p.append(buffer[j][0] - buffer[j][1])
            labels.append(buffer1[j])

    return fixed_p, labels


def clp(cep, mu, w, sigma, cst):
    # Check dimension.
    if cep.ndim == 1:
        cep = cep[numpy.newaxis, :]
    else:
        pass

    # Compute the data independent term for map.
    a = (numpy.square(mu.reshape(mu.shape)) * sigma).sum(1) - 2.0 * (
            numpy.log(w) + numpy.log(cst))

    # Compute the data independent term.
    b = numpy.dot(numpy.square(cep), sigma.T) - 2.0 * numpy.dot(
        cep, numpy.transpose(mu.reshape(mu.shape) * sigma))

    # Compute and return the exponential term
    return -0.5 * (a + b)


def log_likelihood_ratio(ubm, spoof, spoof_file, genuine, genuine_file, dev,
                         h5folder):
    # Read model files for genuine and spoof data
    spoof.read(spoof_file)
    genuine.read(genuine_file)

    ubm.server.feature_filename_structure = h5folder
    # Compute log-likelihood ratio.
    llr_values = []
    lp = []
    for j in range(0, len(dev.file_list)):
        cep, _ = ubm.server.load(dev.file_list[j])

        if genuine.invcov.ndim == 2:
            lp = genuine.compute_log_posterior_probabilities(cep)
        elif genuine.invcov.ndim == 3:
            lp = genuine.compute_log_posterior_probabilities_full(cep)

        ppMax = np.max(lp, axis=1)
        loglk1 = ppMax + np.log(
            np.sum(np.exp((lp.transpose() - ppMax).transpose()), axis=1))

        lp = clp(cep, spoof.mu, spoof.w, spoof.invcov, spoof.cst)
        ppMax = np.max(lp, axis=1)
        loglk2 = ppMax + np.log(
            np.sum(np.exp((lp.transpose() - ppMax).transpose()), axis=1))

        llr_values.append(loglk1.mean() - loglk2.mean())

    # Finally, return log-likelihood ratio.
    return llr_values
