"""
The module provides utility functions for processing score files and calculating
detection performance metrics such as equal error rate (EER) and creating
detection error tradeoff (DET) plots. It also includes functions for calculating
log-likelihood ratios and probability scores.
"""

import math
from typing import List, Tuple

import numpy
from detplot import DetPlot
from sidekit.bosaris.detplot import rocch, rocch2eer


def create_score_file(values: numpy.ndarray, labels: str,
                      file_list: list = None,
                      indexes: list = None) -> List[str]:
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
    for value in values:
        try:
            a = math.log(value[1])
            b = math.log(value[0])
        except ValueError:
            a = math.log(value[1] + eps)
            b = math.log(value[0] + eps)
        llr_values.append(a - b)

    # Frame-wise classification.
    if indexes is not None:
        llr_values = mean_of_frames(llr_values, indexes)

    # Divide scores into tar and non-tar categories.
    tar = []
    nontar = []
    for i, label in enumerate(labels):
        if label in [1, 'genuine']:
            tar.append(str(llr_values[i]))
        elif label in [0, 'spoof']:
            nontar.append(str(llr_values[i]))
        else:
            raise AssertionError('Some of the labels are missing.')

    # Creates score file(filename-label-score) and check if there are missing
    # values.
    if len(llr_values) != len(labels):
        raise AssertionError('Labels and scores are not in same length.')

    # Create score file.
    score_file = []
    for i, llr_value in enumerate(llr_values):
        label_as_string = 'genuine' if str(labels[i]) == '1' else 'spoof'
        score_file.append(f'{file_list[i]} {label_as_string} {llr_value}')

    return score_file


def mean_of_frames(values: numpy.ndarray, indexes: List[int]) -> List[float]:
    """
    Finds the mean of frames (for frame-wise classification).

    Args:
        values: One-dimensional array of values.
        indexes: List of mean indexes.

    Returns:
        List of mean values calculated for each frame.
    """
    means = []
    end = 0
    for index in indexes:
        start = end
        end = start + index
        means.append(numpy.mean(values[start:end]))

    return means


def fix_scores(probabilities: List[Tuple]) -> Tuple[List[float], List[str]]:
    """
    Processes a list of probabilities and calculates the difference between
    each probability pair. Also extracts the corresponding labels.

    Args:
        probabilities: List of tuples containing probability pairs and labels.

    Returns:
        A tuple containing a list of fixed probabilities and a list of labels.
    """
    fixed_p = []
    labels = []
    for probability in probabilities:
        buffer = probability[0]
        buffer1 = probability[1]
        for j in range(len(buffer)):
            fixed_p.append(buffer[j][0] - buffer[j][1])
            labels.append(buffer1[j])

    return fixed_p, labels


def calculate_eer(data: numpy.ndarray, tar: List[float] = None,
                  nontar: List[float] = None) -> float:
    """
    Calculates equal error rate from given file of predictions stored.

    Args:
        data: Score data in --filename-label-score-- format.
        nontar: Non target scores.
        tar: Target scores.

    Returns:
        Rounded (2) equal error rate.
    """
    if tar is None and nontar is None:
        # Extract target and non-target scores.
        tar = []
        nontar = []
        for line in range(len(data)):
            if data[(line, 1)] in ['genuine', '1']:
                tar.append(data[(line, 2)])
            elif data[(line, 1)] in ['spoof', '0']:
                nontar.append(data[(line, 2)])
            else:
                raise AssertionError('Some of the values are missing!')

    # Calculate equal error rate (eer).
    pmiss, pfa = rocch(numpy.asarray(tar), numpy.asarray(nontar))
    eer = rocch2eer(pmiss, pfa) * 100

    return round(eer, 2)


def rocch2eer_custom(pmiss: numpy.ndarray, pfa: numpy.ndarray) -> float:
    """
    Calculates the Equal Error Rate (EER) based on the given miss probabilities
    and false-alarm probabilities.

    Args:
        pmiss: A NumPy array of miss probabilities.
        pfa: A NumPy array of false-alarm probabilities.

    Returns:
        The Equal Error Rate (EER)
    """
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), 'Sort: Pmiss and PFA'

        xy = numpy.column_stack((xx, yy))
        dd = numpy.dot(numpy.array([1, -1]), xy)
        if numpy.min(numpy.abs(dd)) == 0:
            eer_seg = 0
        else:
            seg = numpy.linalg.solve(xy, numpy.array([[1], [1]]))
            eer_seg = 1 / (numpy.sum(seg))

        eer = max([eer, eer_seg])

    return eer


def tar_nontar(score_data: numpy.ndarray) -> Tuple[List[float], List[float]]:
    """
    Extract non-target and target values from a score file.

    Args:
        score_file: Score data.

    Returns:
        A tuple containing lists of non-target and target scores.
    """
    tar = []
    nontar = []
    for i in range(len(score_data)):
        if score_data[(i, 1)] in ['genuine', '1']:
            tar.append(score_data[(i, 2)])
        elif score_data[(i, 1)] in ['spoof', '0']:
            nontar.append(score_data[(i, 2)])
        else:
            raise AssertionError('Some of the values are missing!')

    # Return non-target and target scores.
    return nontar, tar


def det_plot(tar: List[float], non_tar: List[float], sys_name: str = '') -> None:
    """
    Plots a Detection Error Tradeoff (DET) curve using target and non-target
    scores.

    Args:
        tar: List of target scores.
        non_tar: List of non-target scores.
        sys_name: Optional string for the system name (default: empty string).
    """
    # Convert to numpy arrays.
    tar = numpy.asarray(tar)
    non_tar = numpy.asarray(non_tar)

    # Plot the DET curve.
    dp = DetPlot(plot_title='')
    dp.set_system(tar, non_tar, sys_name=sys_name)
    dp.create_figure()
    dp.plot_rocch_det(0)


def clp(cep: object, mu: object, w: object,
        sigma: numpy.ndarray, cst: numpy.ndarray) -> numpy.ndarray:
    """
    Calculates the log-probability of a cepstral feature vector using a
    Gaussian Mixture Model (GMM) with given mean, weight, and covariance
    parameters.

    Args:
        cep: Cepstral feature vector.
        mu: Mean vector of the GMM.
        w: Weight vector of the GMM.
        sigma: Covariance matrix of the GMM.
        cst: Constant factor for GMM.

    Returns:
        Log-probability of the cepstral feature vector.
    """
    # Check dimension.
    if cep.ndim == 1:
        cep = cep[numpy.newaxis, :]

    # Compute the data independent term for map.
    data_independent_term = (numpy.square(mu.reshape(mu.shape)) * sigma).sum(1)
    - 2.0 * (numpy.log(w) + numpy.log(cst))

    # Compute the data independent term.
    data_dependent_term = numpy.dot(numpy.square(cep), sigma.T)
    - 2.0 * numpy.dot(cep, numpy.transpose(mu.reshape(mu.shape) * sigma))

    # Compute and return the exponential term.
    return -0.5 * (data_independent_term + data_dependent_term)


def log_likelihood_ratio(ubm: object, spoof: object, spoof_file: str,
                         genuine: str, genuine_file: str, dev: object,
                         h5folder: str) -> List[float]:
    """
    Calculates the log-likelihood ratio for a set of development files given a
    Universal Background Model (UBM), genuine and spoof models, and their
    respective files.

    Args:
        ubm: Universal Background Model object.
        spoof: Spoof model object.
        spoof_file: Spoof model file path.
        genuine: Genuine model object.
        genuine_file: Genuine model file path.
        dev: Development dataset object containing file_list attribute.
        h5folder: Folder containing feature files.

    Returns:
        A list of log-likelihood ratios for the development files.
    """
    # Read model files for genuine and spoof data.
    spoof.read(spoof_file)
    genuine.read(genuine_file)

    ubm.server.feature_filename_structure = h5folder

    # Compute log-likelihood ratio.
    llr_values = []
    lp = []
    for j in range(len(dev.file_list)):
        cep, _ = ubm.server.load(dev.file_list[j])

        if genuine.invcov.ndim == 2:
            lp = genuine.compute_log_posterior_probabilities(cep)
        elif genuine.invcov.ndim == 3:
            lp = genuine.compute_log_posterior_probabilities_full(cep)

        ppMax = numpy.max(lp, axis=1)
        loglk1 = ppMax + numpy.log(
            numpy.sum(numpy.exp((lp.transpose() - ppMax).transpose()), axis=1))

        lp = clp(cep, spoof.mu, spoof.w, spoof.invcov, spoof.cst)
        ppMax = numpy.max(lp, axis=1)
        loglk2 = ppMax + numpy.log(
            numpy.sum(numpy.exp((lp.transpose() - ppMax).transpose()), axis=1))

        llr_values.append(loglk1.mean() - loglk2.mean())

    return llr_values
