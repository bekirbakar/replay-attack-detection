import pickle
import timeit

import matplotlib.pyplot as plt
import numpy
import sidekit
import theano
import theano.tensor as t
import theano.tensor as T
from scoring import calculate_eer, load_data

# These codes adapted from lisa-lab tutorials/repository.
# https://github.com/lisa-lab/DeepLearningTutorials/tree/master/code
# http://deeplearning.net/tutorial/


class UBM(object):
    def __init__(self, obj):
        self.ubm_genuine_list = obj.genuine_list
        self.ubm_spoof_list = obj.spoof_list
        self.genuine = sidekit.Mixture()
        self.spoof = sidekit.Mixture()
        self.path = '/SIDEKIT/'
        self.server = sidekit.FeaturesServer(
            features_extractor=None,
            feature_filename_structure=f'{self.path}train/' + '{}.h5',
            sources=None,
            dataset_list=['cep'],
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
            keep_all_features=True,
        )

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
        self.genuine.write(f'{self.path}ubm/genuine.h5')

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
        self.spoof.write(f'{self.path}ubm/spoof.h5')


class LogisticRegression(object):
    def __init__(self, _input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b',
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
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
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

            w = theano.shared(value=w_values, name='w', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

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
        self.hiddenLayers.extend(
            HiddenLayer(
                rng=rng,
                _input=self.hiddenLayers[i - 1].output,
                n_in=n_hidden[i - 1],
                n_out=n_hidden[i],
                activation=t.tanh,
            )
            for i in range(1, n_hidden_layers)
        )
        self.logRegressionLayer = \
            LogisticRegression(_input=self.hiddenLayers[-1].output,
                               n_in=n_hidden[-1],
                               n_out=n_out)

        self.L1 = (
            sum(abs(h1.w).sum() for h1 in self.hiddenLayers)
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            sum((h1.w**2).sum() for h1 in self.hiddenLayers)
            + (self.logRegressionLayer.W**2).sum()
        )

        self.negative_log_likelihood = \
            self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors
        self.params = self.logRegressionLayer.params

        for hl in self.hiddenLayers:
            self.params += hl.params

        self.input = _input
        self.model_prediction = self.logRegressionLayer.model_prediction


def run_experiment():
    learning_rate = 0.001
    L1_reg = 0.00
    L2_reg = 0.0001
    n_epochs = 1000
    dataset = 'path-to-feature.pkl.gz'
    batch_size = 500
    # n_hidden = 512

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, _ = datasets[1]

    # Compute number of mini-batches for training, validation and testing.
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    # Build Model
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(rng=rng, input_=x, n_in=512, n_hidden=512, n_out=2)

    cost = (classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.probabilities(y),
        givens={
            x: valid_set_x[index * n_test_batches:(index + 1) * n_test_batches],
            y: valid_set_y[index * n_test_batches:(index + 1) * n_test_batches]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * n_test_batches: (index + 1) * n_test_batches],
            y: train_set_y[index *
                           n_test_batches: (index + 1) * n_test_batches]
        }
    )

    # Train Model

    # Early-stopping Parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    eer = []

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):

            train_model(minibatch_index)
            iteration = (epoch - 1) * n_train_batches + minibatch_index

            if (iteration + 1) % validation_frequency == 0:
                # Compute zero-one loss on validation set.
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]

                this_validation_loss = numpy.mean(validation_losses)
                # _, this_eer = score('valid.txt', validation_losses)
                # print('This EER = {}  for epoch{}\n'.format(this_eer, epoch))

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # If there is the best validation score until now.
                if this_validation_loss < best_validation_loss:
                    # Improve patience if loss improvement is good enough.
                    if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                        patience = max(patience, iteration * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iteration

                    # Test
                    p_values = [test_model(i) for i in range(1)]
                    this_eer = calculate_eer(p_values)
                    eer.append(this_eer)

                    print(f'Best eer until now is ___{this_eer}____')
                    print(('Epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # Save the best model.
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iteration:
                done_looping = True
                break

    plt.plot(eer)
