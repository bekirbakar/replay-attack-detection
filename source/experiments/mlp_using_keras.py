# -*- coding: utf-8 -*-
import keras
import numpy
from keras.backend import clear_session
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential

import source.scoring as scoring
from source.data_io_utils import load_excel, read_data, save_data, save_excel


def create_network(n_dense, input_shape, dense_units, activation, dropout, dropout_rate,
                   kernel_initializer='lecun_normal', optimizer='sgd'):
    """
    Generic function to create a fully-connected neural network.

    Args:
        n_dense: int > 0. Number of dense layers.
        input_shape: ?
        dense_units: dense_units: int > 0. Number of dense units per layer.
        activation: ?
        dropout: keras.layers.Layer. A dropout layer to apply.
        dropout_rate: 0 <= float <= 1. The rate of dropout.
        kernel_initializer: str. The initializer for the weights.
        optimizer: str/keras.optimizers.Optimizer. The optimizer to use.
        # num_classes: int > 0. The number of classes to predict.
        # max_words: int > 0. The maximum number of words per data point.

    Returns:
        A Keras model instance (compiled).
    """
    # Clear gpu memory first.
    clear_session()

    # Set input layer.
    model = Sequential()
    model.add(Dense(dense_units, input_shape=(input_shape,), kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

    # Set hidden layer(s).
    for _ in range(n_dense - 1):
        model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        # model.add(BatchNormalization())
        model.add(dropout(dropout_rate))

    # Set output layer.
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Print out model summary.
    model.summary()

    return model


def evaluate_model(filename, e_data):
    """
    Performs an evaluation of best model on data.

    Args:
        filename: Evaluation data.
        e_data: Best model index, starts from 0.

    Returns:
        EER and score_file.
    """
    # Load best model.
    path_to_model = f'../data/models/{filename}.h5'
    best_model = keras.models.load_model(path_to_model)

    return best_model.predict(e_data, verbose=0)


# Feature list for loop.
feature_list = ['ltas']
evaluate = True
for feature_ in feature_list:
    print(f'Reading {feature_} data...')

    # Read mlp parameters from excel file.
    mlp_params = load_excel(feature_)

    # Read training data.
    train_data = read_data(feature_, 'train')
    x_train = train_data['x_data']
    y_train = train_data['y_data']
    batch_size = train_data['mini_batch_size']

    # Read development data and labels.
    test_data = read_data(feature_, 'dev')
    x_test = test_data['x_data']
    y_test = test_data['y_data']
    test_labels = test_data['labels']
    test_file_list = test_data['file_list']
    test_indexes = test_data['indexes']

    # Delete data to release memory.
    del train_data, test_data

    # Loop through rows in mlp_params.
    for line in range(len(mlp_params)):
        # Create network.
        print('Creating network...')
        network = {'n_dense': len(mlp_params['dense_units'][line].split('-')),
                   'dense_units': int(mlp_params['dense_units'][line].split('-')[0]),
                   'activation': mlp_params['activation'][line],
                   'dropout': Dropout,
                   'dropout_rate': 0.4,
                   'optimizer': 'sgd'}
        # Create model.
        keras_model = create_network(input_shape=len(x_train[0]), **network)

        # Placeholders to store metrics.
        val_acc = []
        dev_eer = []
        best_eer = numpy.inf

        # Early stopping parameters.
        early_stop = 0
        done_looping = False
        epoch = 0
        while (epoch < 10000) and (not done_looping):
            # Report '1' for first epoch, 'n_epochs' for last epoch.
            epoch += 1

            # Train model.
            metrics = keras_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1,
                                      batch_size=batch_size, shuffle=True, verbose=0)
            predictions = keras_model.predict(x_test, verbose=0, batch_size=batch_size)

            score_file = scoring.create(probabilities=predictions, labels=test_labels,
                                        file_list=test_file_list, indexes=test_indexes)
            this_eer = scoring.calculate_eer(score_file)
            dev_eer.append(this_eer)
            val_acc.append(metrics.history['val_acc'][0] * 20)
            print('Epoch = {}, Eer (validation) = {}, accuracy(train) = {:,.4}'.format(epoch, this_eer,
                                                                                       metrics.history['acc'][0]))
            early_stop += 1
            if this_eer < best_eer * 0.995:
                # If best eer is found.
                best_eer = this_eer
                save_data(score_file, (feature_ + str(line) + 'dev.txt'))

                # Save best model.
                model_file = f'../data/models/best_model_{feature_}{str(line)}.h5'
                keras_model.save(model_file)
                print(f'Best eer = {best_eer}, model saved.\n')
                early_stop = 0

            if early_stop == 300:
                done_looping = True
                break

        if evaluate:
            print('Loading evaluation data...')
            eval_data = read_data(feature_, 'eval')
            load_eval_data = False
            eval_eer = evaluate_model(eval_data, line)['eval_eer']
        else:
            eval_eer = 0
        save_excel('mlp-scores.xls', feature_, line)

# Evaluate model.
ltas_results = []
eval_data = ''  # Read data here. Example: read_data('mfcc', 'eval')
for index in range(1):
    evaluation = evaluate_model(eval_data, index)
    ltas_results.append(evaluation['eval_eer'])
