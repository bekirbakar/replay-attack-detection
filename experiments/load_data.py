# -*- coding: utf-8 -*-
import gzip
import pickle

import numpy as np
import theano
import theano.tensor as t


def load_data(dataset):
    try:
        with gzip.open(dataset, "rb") as f:
            train, dev, eva = pickle.load(f, encoding="latin1")
            f.close()
    except ValueError or EOFError:
        with gzip.open(dataset, "rb") as f:
            train, dev = pickle.load(f, encoding="latin1")
            f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, t.cast(shared_y, "int32")

    train_x, train_y = shared_dataset(train)
    dev_x, dev_y = shared_dataset(dev)

    rval = [(train_x, train_y), (dev_x, dev_y)]

    return rval
