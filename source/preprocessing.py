# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal

from helpers import flush_process_info


def split_frames(feats, labels):
    """
    Splits given ceps into an array list. One element of feats lists is stored
    in mxn numpy.ndarray (n is ceps number), the function store this data into
    nx1 numpy.ndarray.

    Args:
        feats: Feature list, every list element must contain ndarray.
        labels: Corresponding label of feature.

    Returns:
        A tuple of features, labels and dimension of each list elements.
    """
    # Get indexes, length of every mXn dimensional list element.
    indexes = []
    for i in range(0, len(feats)):
        indexes.append(feats[i].shape[0])

    # Declare x, y to store features and labels, respectively.
    x = []
    y = []
    print("Splitting!")

    j = 0
    for i in range(0, len(feats)):
        buffer = feats[i]

        # Append data into x and labels into y.
        j: int
        for j in range(0, len(buffer)):
            x.append(buffer[j])
            y.append(labels[i])
        j += 1

    return tuple((x, y, indexes))


def split_ceps(feats, labels):
    """
    Splits given ceps into an array list. One element of feats lists is stored
    in mxn numpy.ndarray(n is ceps number), the function store this data into
    nx1 numpy.ndarray.

    Args:
        feats: Feature list, every list element must contain numpy.ndarray.
        labels: Corresponding label of feature.

    Returns:
        A tuple for features, labels and dimension info of each list elements.
    """
    # Get info, length of every mxn dimensional list element.
    count = 0
    info = []
    for i in range(0, len(feats)):
        info.append(feats[i].shape[0])
        count += info[i]

    # Calculate index for each list and store according to index into x and y.
    x = []  # Declare x to store features.
    y = []  # Declare y to store labels.
    index = 0
    j = 0
    for i in range(0, len(feats)):
        # Store data into buffer.
        index += j
        buffer = feats[i]

        # Append data into x.
        for j in range(0, len(buffer)):
            x.append(buffer[j])
            y.append(labels[j])
        j += 1
        flush_process_info("Completed.", i, len(feats))

    return x, y, info


def mvn(feat):
    """
    Performs mean and variance normalization.

    Args:
        feat: Data to be normalized.

    Returns:
        Normalized data.
    """
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
        for i in range(0, feat.shape[1]):
            mu = np.mean(feat[:, i])
            sigma = np.std(feat[:, i])
            norm_feat[:, i] = (feat[:, i] - mu) / sigma

    return norm_feat


def cmvn(feat):
    """
    Performs cepstral mean and variance normalization.

    Args:
        feat: Data to be normalized.

    Returns:
        Normalized data.
    """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Find mean and variance of one dimensional features.
    mu = np.mean(feat)
    sigma = np.std(feat) + eps

    # Subtract mu from each element and divide by sigma.
    feat -= mu
    feat /= sigma + 2.2204e-16

    # Check "nan" value existence.
    if np.isnan(feat).any() is np.bool_(True):
        raise AssertionError("None value encountered.")

    return feat


def cvn(feat):
    """
    Performs variance normalization.

    Args:
        feat: Input features.
    Returns:
        Output features.
    """
    eps = 2.2204e-16  # Avoid dividing values to zero.
    sigma = np.std(feat) + eps  # Find sigma.
    feat /= sigma  # Divide by sigma.

    # Check "nan" value existence.
    if np.isnan(feat).any() is np.bool_(True):
        raise AssertionError("None (nan) value encountered.")

    return feat


def cms(feat):
    """
    Performs cepstral mean subtraction.

    Args:
        feat: Input features.

    Returns:
        Output features.
    """
    # Find mean and subtract mu from each element.
    mu = np.mean(feat)
    feat -= mu

    return feat


class Filter:
    def __init__(self, x, f_cut=3000, order=10):
        """
        Applies high pass filter to signal.

        Args:
            x: Input signal.
            f_cut: Cut frequency.
            order: Filtered signal.
        """
        self.x = x
        self.f_cut = f_cut
        self.order = order

    def high_pass_filter(self):
        """
        Applies high pass filter to signal.

        Returns:
            Filtered signal.
        """
        # Design a high-pass filter.
        f_cut = self.f_cut / 8000
        [b, a] = signal.butter(self.order, f_cut, "high")

        # Apply filter to signal.
        x = signal.lfilter(b, a, self.x)

        return x
