# -*- coding: utf-8 -*-
import numpy
import sidekit
import theano
import theano.tensor as t


# These codes adapted from lisa-lab tutorials/repository.
# https://github.com/lisa-lab/DeepLearningTutorials/tree/master/code
# http://deeplearning.net/tutorial/

class UBM(object):
    def __init__(self, obj):
        self.ubm_genuine_list = obj.genuine_list
        self.ubm_spoof_list = obj.spoof_list
        self.genuine = sidekit.Mixture()
        self.spoof = sidekit.Mixture()
        self.path = "/SIDEKIT/"
        self.server = sidekit.FeaturesServer(
            features_extractor=None,
            feature_filename_structure=self.path + "train/" + "{}.h5",
            sources=None,
            dataset_list=["cep"],
            mask=None,
            feat_norm=None,
            global_cmvn=None,
            dct_pca=False,
            dct_pca_config=None,
            sdc=False,
            sdc_config=None,
            delta=True,
            double_delta=True,
            delta_filter=None,
            context=None,
            traps_dct_nb=None,
            rasta=False,
            keep_all_features=True)

    def split(self):
        self.genuine.EM_split(
            features_server=self.server,
            feature_list=self.ubm_genuine_list,
            distrib_nb=1024,
            iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
            llk_gain=0.01,
            num_thread=1,
            save_partial=True,
            ceil_cov=10,
            floor_cov=1e-2)
        self.genuine.write(self.path + "ubm/genuine.h5")

        self.spoof.EM_split(
            features_server=self.server,
            feature_list=self.ubm_spoof_list,
            distrib_nb=1024,
            iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
            llk_gain=0.01,
            num_thread=1,
            save_partial=True,
            ceil_cov=10,
            floor_cov=1e-2)
        self.spoof.write(self.path + "ubm/spoof.h5")


class LogisticRegression(object):
    def __init__(self, _input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name="W",
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name="b",
                               borrow=True)
        self.p_y_given_x = t.nnet.softmax(t.dot(_input, self.W) + self.b)
        self.y_pred = t.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = _input

    def model_prediction(self, y):
        return self.p_y_given_x, y

    def negative_log_likelihood(self, y):
        return -t.mean(t.log(self.p_y_given_x)[t.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith("int"):
            return t.mean(t.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, _input, n_in, n_out, w=None, b=None,
                 activation=t.tanh):
        self.input = _input

        if w is None:
            w_values = \
                numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                          high=numpy.sqrt(6. / (n_in + n_out)),
                                          size=(n_in, n_out)),
                              dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                w_values *= 4

            w = theano.shared(value=w_values, name="w", borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)

        self.w = w
        self.b = b

        lin_output = t.dot(_input, self.w) + self.b

        self.output = \
            (lin_output if activation is None else activation(lin_output))
        self.params = [self.w, self.b]


class MLP(object):
    def __init__(self, rng, input_, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(rng=rng,
                                       _input=input_,
                                       n_in=n_in,
                                       n_out=n_hidden,
                                       activation=theano.tensor.nnet.relu)

        self.logRegressionLayer = \
            LogisticRegression(_input=self.hiddenLayer.output,
                               n_in=n_hidden,
                               n_out=n_out)

        self.L1 = (abs(self.hiddenLayer.w).sum()
                   + abs(self.logRegressionLayer.W).sum())

        self.L2_sqr = ((self.hiddenLayer.w ** 2).sum()
                       + (self.logRegressionLayer.W ** 2).sum())

        self.negative_log_likelihood = \
            self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input_
        self.model_prediction = self.logRegressionLayer.model_prediction


class DeepMLP(object):
    def __init__(self, rng, _input, n_in, n_hidden, n_out,
                 n_hidden_layers):
        self.hiddenLayers = [HiddenLayer(rng=rng,
                                         _input=_input,
                                         n_in=n_in,
                                         n_out=n_hidden[0],
                                         activation=theano.tensor.nnet.relu)]
        for i in range(1, n_hidden_layers):
            self.hiddenLayers.append(HiddenLayer(rng=rng,
                                                 _input=self.hiddenLayers[
                                                     i - 1].output,
                                                 n_in=n_hidden[i - 1],
                                                 n_out=n_hidden[i],
                                                 activation=t.tanh))

        self.logRegressionLayer = \
            LogisticRegression(_input=self.hiddenLayers[-1].output,
                               n_in=n_hidden[-1],
                               n_out=n_out)

        self.L1 = (sum([abs(h1.w).sum() for h1 in self.hiddenLayers])
                   + abs(self.logRegressionLayer.W).sum())

        self.L2_sqr = (sum([(h1.w ** 2).sum() for h1 in self.hiddenLayers])
                       + (self.logRegressionLayer.W ** 2).sum())

        self.negative_log_likelihood = \
            self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors
        self.params = self.logRegressionLayer.params

        for hl in self.hiddenLayers:
            self.params += hl.params

        self.input = _input
        self.model_prediction = self.logRegressionLayer.model_prediction
