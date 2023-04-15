# -*- coding: utf-8 -*-
import timeit

import numpy
import theano
import theano.tensor as T

import source.features as features
from source.scoring import create, fix_scores, mean_of_frames


class LogisticRegression:
    pass


# Definitions
path_to_mfcc_features = '../data/features/mfcc.pkl.gz'
mfcc = features.Feature.visualize(path_to_mfcc_features)

learning_rate = 0.13
n_epochs = 100
batch_size = 100000

datasets = features.load_features(mfcc)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[1]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0]

INN = 20

index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

classifier = LogisticRegression(input=x, n_in=INN, n_out=2)
cost = classifier.negative_log_likelihood(y)

test_model = theano.function(
    inputs=[index],
    outputs=classifier.model_prediction(y),
    givens={
        x: valid_set_x[index * n_test_batches: (index + 1) * n_test_batches],
        y: valid_set_y[index * n_test_batches: (index + 1) * n_test_batches]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

updates = [(classifier.W, classifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]}
)

patience = 5000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience // 2)

best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()
done_looping = False
epoch = 0

this_validation_loss = []
eer = []
while (epoch < n_epochs) and (not done_looping):
    epoch += 1
    for minibatch_index in range(n_train_batches):
        train_model(minibatch_index)
        iteration = (epoch - 1) * n_train_batches + minibatch_index

        if (iteration + 1) % validation_frequency == 0:
            validation_losses = [validate_model(i) for i in range(n_valid_batches)]
            validation_losses = numpy.mean(validation_losses)
            this_validation_loss = validation_losses

            print('Looping')

            print('Epoch %i, minibatch %i%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iteration * patience_increase)

                best_validation_loss = this_validation_loss

                p_values = [test_model(i) for i in range(1)]

                fixed_scores, _ = fix_scores(p_values)
                mean_of_scores = mean_of_frames(fixed_scores, _)
                score_file, this_eer = create('frame_wise_score.txt', mean_of_scores)

                eer.append(this_eer)

                print(('Epoch %i , minibatch %i%i, test eer of best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches, this_eer))

        if patience <= iteration:
            done_looping = True
            print()
            break
