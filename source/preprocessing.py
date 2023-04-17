"""
This module provides functions for features preprocessing and normalization,
focusing on resizing, mean and variance normalization, and filtering. The
primary purpose of this module is to prepare input data audio processing.

Functions in this module enable you to resize features arrays to a specific
length, fix features dimensions by appending zeros, split features arrays,
perform mean and variance normalization, and apply high-pass filters to input
signals.
"""

import pickle
from typing import Tuple

import numpy as numpy
from data_io_utils import flush_process_info
from scipy import signal


def fix_to_len(features: numpy.ndarray, len_to_fit: int) -> numpy.ndarray:
    """
    Resizes the features arrays to the specified length using random duplication
    of elements.

    Args:
        features: A numpy array of features arrays.
        len_to_fit: The desired length of the output features arrays.

    Returns:
        A numpy array of resized features arrays.
    """
    final_features = []
    v_len = features.shape[1]
    diff = len_to_fit - v_len
    indexes = numpy.random.choice(v_len, diff, replace=True)
    for features in features:
        buffer = numpy.array([features[indexes[index]]
                             for index in range(diff)])
        final_features.append(numpy.append(features, buffer))

    return numpy.asarray(final_features)


def fix_to_max(features: numpy.ndarray) -> numpy.ndarray:
    """
    Resizes the features arrays to the maximum length in the input by
    randomly duplicating elements.

    Args:
        features: A numpy array of features arrays.

    Returns:
        A numpy array of resized features arrays.
    """
    lengths = [len(features.flatten()) for features in features]
    max_len = max(lengths)

    final_features = []
    for item in (features.flatten() for features in features):
        v_len = len(item)
        diff = max_len - v_len
        indexes = numpy.random.choice(v_len, diff, replace=True)
        buffer = numpy.array([item[indexes[index]] for index in range(diff)])
        final_features.append(numpy.append(item, buffer))

    return numpy.asarray(final_features)


def fix_to_min(features: numpy.ndarray) -> numpy.ndarray:
    """
    Trims the features arrays to the minimum length in the input.

    Args:
        features: A numpy array of features arrays.

    Returns:
        A numpy array of resized features arrays.
    """
    lengths = [len(features.flatten()) for features in features]
    min_len = min(lengths)
    final_features = [item[:min_len]
                      for item in (features.flatten() for features in features)]
    return numpy.asarray(final_features)


def fix_features_dtype(data: dict, destination_file: str) -> None:
    """
    Type conversion and features dimension fixing.

    Args:
        data: A dictionary containing 'x_data', 'y_data', 'labels', and
              'file_list'.
        destination_file: The base name of the destination file.
    """
    x_data = data['x_data']
    y_data = data['y_data']
    labels = data['labels']
    file_list = data['file_list']
    del data

    x_data = fix_to_min(x_data.copy())
    x_data = x_data.astype('float32')
    data = {'x_data': x_data, 'y_data': y_data,
            'labels': labels, 'file_list': file_list}

    with open(f'{destination_file}.pkl.gz', 'wb+') as fp:
        pickle.dump(data, fp)


def fix_features_dimensions(
        train: numpy.ndarray, test: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Fixes features dimensions by appending zeros to match the maximum length.

    Args:
        train: A numpy array of training features arrays.
        test: A numpy array of testing features arrays.

    Returns:
        A tuple containing fixed training and testing features arrays.
    """
    train_frame_lens = [len(l) for l in train]
    test_frame_lens = [len(l) for l in test]
    max_len = max(max(train_frame_lens), max(test_frame_lens))

    fixed_train = fix_dimensions(train, max_len)
    fixed_test = fix_dimensions(test, max_len)

    return fixed_train, fixed_test


def fix_dimensions(data: numpy.ndarray, max_len: int) -> numpy.ndarray:
    """
    Fixes features dimensions by appending zeros to match the maximum length.

    Args:
        data: A numpy array of features arrays.
        max_len: The maximum length to resize the features arrays.

    Returns:
        A numpy array of fixed features arrays.
    """
    n_frames = data[0].shape[1]
    fixed_data = numpy.ones([len(data), max_len, n_frames], dtype='float32')
    for index, item in enumerate(data):
        diff = max_len - item.shape[0]
        zeros = numpy.zeros((diff, n_frames), dtype='float32')
        fixed_data[index] = numpy.append(item, zeros, axis=0)

    return fixed_data


def split_frames(features: list, labels: list) -> Tuple[list, list]:
    """
    Splits given ceps into an array list. One element of features lists is stored
    in mxn numpy.ndarray (n is ceps number), the function store this data into
    nx1 numpy.ndarray.

    Args:
        features: features list, every list element must contain ndarray.
        labels: Corresponding label of features.

    Returns:
        A tuple of features, labels and dimension of each list elements.
    """
    indexes = [features[i].shape[0] for i in range(len(features))]
    # Declare x, y to store features and labels, respectively.
    x = []
    y = []
    print('Splitting!')

    j = 0
    for i in range(len(features)):
        buffer = features[i]

        # Append data into x and labels into y.
        j: int
        for j in range(len(buffer)):
            x.append(buffer[j])
            y.append(labels[i])
        j += 1

    return x, y, indexes


def split_ceps(features: list, labels: list) -> Tuple[list, list]:
    """
    Splits given ceps into an array list. One element of features lists is
    stored in mxn numpy.ndarray(n is ceps number), the function store this data
    into nx1 numpy.ndarray.

    Args:
        features: features list, every list element must contain numpy.ndarray.
        labels: Corresponding label of features.

    Returns:
        A tuple for features, labels and dimension info of each list elements.
    """
    # Get info, length of every mxn dimensional list element.
    count = 0
    info = []
    for i in range(len(features)):
        info.append(features[i].shape[0])
        count += info[i]

    # Calculate index for each list and store according to index into x and y.
    x = []  # Declare x to store features.
    y = []  # Declare y to store labels.
    index = 0
    j = 0
    for i in range(len(features)):
        # Store data into buffer.
        index += j
        buffer = features[i]

        # Append data into x.
        for j in range(len(buffer)):
            x.append(buffer[j])
            y.append(labels[j])
        j += 1
        flush_process_info('Completed.', i, len(features))

    return x, y, info


def mvn(features: numpy.ndarray):
    """
    Performs mean and variance normalization.

    Args:
        features: Data to be normalized.

    Returns:
        Normalized data.
    """
    if len(features.shape) == 1:
        normalized_features = features.copy()

        # Find mean and variance.
        mu = numpy.mean(features)
        sigma = numpy.std(features)
        normalized_features -= mu
        normalized_features /= sigma
    else:
        normalized_features = numpy.zeros([features.shape[0],
                                           features.shape[1]], dtype=float)
        for i in range(features.shape[1]):
            mu = numpy.mean(features[:, i])
            sigma = numpy.std(features[:, i])
            normalized_features[:, i] = (features[:, i] - mu) / sigma

    return normalized_features


def cmvn(features: numpy.ndarray):
    """
    Performs cepstral mean and variance normalization.

    Args:
        features: Data to be normalized.

    Returns:
        Normalized data.
    """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Find mean and variance of one dimensional features.
    mu = numpy.mean(features)
    sigma = numpy.std(features) + eps

    # Subtract mu from each element and divide by sigma.
    features -= mu
    features /= sigma + 2.2204e-16

    # Check 'nan' value existence.
    if numpy.isnan(features).any() is numpy.bool_(True):
        raise AssertionError('None value encountered.')

    return features


def cvn(features: numpy.ndarray):
    """
    Performs variance normalization.

    Args:
        features: Input features.
    Returns:
        Output features.
    """
    eps = 2.2204e-16  # Avoid dividing values to zero.
    sigma = numpy.std(features) + eps  # Find sigma.
    features /= sigma  # Divide by sigma.

    # Check 'nan' value existence.
    if numpy.isnan(features).any() is numpy.bool_(True):
        raise AssertionError('None (nan) value encountered.')

    return features


def cms(features: numpy.ndarray):
    """
    Performs cepstral mean subtraction.

    Args:
        features: Input features.

    Returns:
        Output features.
    """
    # Find mean and subtract mu from each element.
    mu = numpy.mean(features)
    features -= mu

    return features


def high_pass_filter(x: numpy.ndarray, f_cut: int = 3000,
                     order: int = 10) -> numpy.ndarray:
    """
    Applies high pass filter to signal.

    Args:
        x: Input signal.
        f_cut: Cut frequency.
        order: Filtered signal.

    Returns:
        Filtered signal.
    """
    # Design a high-pass filter.
    f_cut_normalized = f_cut / 8000
    b, a = signal.butter(order, f_cut_normalized, 'high')

    return signal.lfilter(b, a, x)
