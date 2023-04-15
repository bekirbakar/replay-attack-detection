# -*- coding: utf-8 -*-
import pickle
import timeit

import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as T
from theano_models import MLP

from source.load_data import load_data
from source.scoring import calculate_eer

learning_rate = 0.001
L1_reg = 0.00
L2_reg = 0.0001
n_epochs = 1000
dataset = "path-to-feature.pkl.gz"
batch_size = 500
n_hidden = 512

datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[1]

# Compute number of mini-batches for training, validation and testing.
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0]

# Build Model
index = T.lscalar()
x = T.matrix("x")
y = T.ivector("y")

rng = numpy.random.RandomState(1234)

classifier = MLP(rng=rng, input_=x, n_in=512, n_hidden=512, n_out=2)

cost = (classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr)

# noinspection PyUnresolvedReferences
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
        y: train_set_y[index * n_test_batches: (index + 1) * n_test_batches]
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
            # _, this_eer = score("valid.txt", validation_losses)
            # print("This EER = {}  for epoch{}\n".format(this_eer, epoch))

            print("epoch %i, minibatch %i/%i, validation error %f %%" %
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

                print(f"Best eer until now is ___{this_eer}____")
                print(("Epoch %i, minibatch %i/%i, test error of "
                       "best model %f %%") %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

                # Save the best model.
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(classifier, f)

        if patience <= iteration:
            done_looping = True
            break

plt.plot(eer)
