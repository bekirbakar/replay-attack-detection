# -*- coding: utf-8 -*-
import pickle

import numpy


def fix_to_len(feats, len_to_fit):
    final_feats = []
    v_len = feats.shape[1]
    diff = len_to_fit - v_len
    indexes = numpy.random.choice(v_len, diff, replace=True)
    for feat in range(0, len(feats)):
        buffer = []
        for index in range(0, diff):
            buffer.append(feats[feat][indexes[index]])
        buffer = numpy.asarray(buffer.copy())
        final_feats.append(numpy.append(feats[feat], buffer))

    return numpy.asarray(final_feats)


def fix_to_max(feats):
    # Store feats in list as vector and keep feat lengths.
    lengths = []
    feats_as_vector = []
    for feat in range(0, len(feats)):
        feats_as_vector.append((feats[feat]).flatten())
        lengths.append(len(feats_as_vector[feat]))
    max_len = max(lengths)
    final_feats = []
    for feat in range(0, len(feats_as_vector)):
        v_len = len(feats_as_vector[feat])
        diff = max_len - v_len
        indexes = numpy.random.choice(v_len, diff, replace=True)
        buffer = []
        for index in range(0, diff):
            buffer.append(feats_as_vector[feat][indexes[index]])
        buffer = numpy.asarray(buffer.copy())
        final_feats.append(numpy.append(feats_as_vector[feat], buffer))

    return numpy.asarray(final_feats)


def fix_to_min(feats):
    # Store feats in list as vector and keep feat lengths.
    lengths = []
    feats_as_vector = []
    for feat in range(0, len(feats)):
        feats_as_vector.append((feats[feat]).flatten())
        lengths.append(len(feats_as_vector[feat]))
    min_len = min(lengths)
    final_feats = []
    for feat in range(0, len(feats_as_vector)):
        final_feats.append(feats_as_vector[feat][0:min_len])

    return numpy.asarray(final_feats)


def fix_feats(data, destination_file):
    x_data = data["x_data"]
    y_data = data["y_data"]
    labels = data["labels"]
    file_list = data["file_list"]
    del data

    # x_data = fix_to_max(x_data.copy())
    x_data = fix_to_min(x_data.copy())
    x_data = x_data.astype("float32")
    data = {"x_data": x_data, "y_data": y_data, "labels": labels,
            "file_list": file_list}

    with open(destination_file + ".pkl.gz", "wb+") as fp:
        pickle.dump(data, fp)


def fix_feat_dims(train, test):
    # Frame lens.
    train_frame_lens = [len(l) for l in train]
    test_frame_lens = [len(l) for l in test]

    # Max len.
    max_len = max(max(train_frame_lens), max(test_frame_lens))

    # Train.
    n_frames = train[0].shape[1]
    fixed_train = numpy.ones([len(train), max_len, n_frames], dtype="float32")
    for index in range(0, len(train)):
        diff = max_len - train[index].shape[0]
        zeros = numpy.zeros((diff, n_frames), dtype="float32")
        fixed_train[index] = numpy.append(train[index], zeros, axis=0)

    # Dev.
    print("\nTest features.")
    n_frames = test[0].shape[1]
    fixed_test = numpy.ones([len(test), max_len, n_frames], dtype="float32")
    for index in range(0, len(test)):
        diff = max_len - test[index].shape[0]
        zeros = numpy.zeros((diff, n_frames), dtype="float32")
        fixed_test[index] = numpy.append(test[index], zeros, axis=0)

    return fixed_train, fixed_test
