import json
import pickle

import keras
import numpy
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from source.asvspoof_data import ASVspoof
from source.data_io_utils import load_excel, save_excel
from source.scoring import save, scoring

with open('./config/datasets.json') as fh:
    dataset_config = json.loads(fh.read())


def create_network(n_dense,
                   input_shape,
                   dense_units,
                   activation,
                   dropout,
                   dropout_rate,
                   optimizer='sgd'):
    '''
    Generic function to create a fully-connected neural network.
    '''
    # Clear gpu memory first.
    keras.backend.clear_session()

    # Set input layer.
    model = Sequential()
    model.add(Dense(dense_units, input_shape=(input_shape,),
                    kernel_initializer='lecun_normal', use_bias=False))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(dropout(dropout_rate))

    # Set hidden layer(s).
    for i in range(n_dense - 1):
        model.add(Dense(dense_units, kernel_initializer='lecun_normal',
                        use_bias=False))
        model.add(Activation(activation))
        model.add(BatchNormalization())
        model.add(dropout(dropout_rate))

    # Set output layer.
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Compile model.
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    # Print out model summary.
    model.summary()

    return model


def read_data(dataset, subset):
    '''
    Reads data from file using joblib.

    :param dataset: filename to be read.
    :param subset: dataset partition, can be 'train', 'dev', 'eval'.
    return data: a dictionary of dataset, indexes, x_data, y_data, labels
    and batch size.
    '''
    # Read data.
    path = f'feats/{dataset.upper()}/'
    file = open(path + dataset + '_' + subset, 'rb')
    data = pickle.load(file)
    # Convert to numpy array.
    x_data = numpy.asarray(data[0])
    x_data = x_data.astype('float32')
    y_data = numpy.asarray(data[1])
    y_data = keras.utils.to_categorical(y_data, 2)
    y_data = y_data.astype('float32')
    # Get labels and file_list.
    dataset_instance = ASVspoof(
        2017,
        subset,
        dataset_config['ASVspoof 2017'][subset]['path_to_dataset'],
        dataset_config['ASVspoof 2017'][subset]['path_to_protocol'],
        dataset_config['ASVspoof 2017'][subset]['path_to_wav']

    )
    labels = dataset_instance.labels
    file_list = dataset_instance.file_list
    # Frame wise classification.
    if dataset in ['mfcc', 'cqcc']:
        indexes = data[2]
    elif dataset == 'ltas':
        indexes = None
    else:
        raise AssertionError('Dataset not found.')
    mini_batch_size = len(
        x_data) // 50 if len(x_data) >= 10000 else len(x_data)
    del data

    # Store values in data dictionary.
    return {
        'dataset': dataset,
        'indexes': indexes,
        'x_data': x_data,
        'y_data': y_data,
        'labels': labels,
        'file_list': file_list,
        'mini_batch_size': mini_batch_size
    }


def evaluate_model(data, index_):
    '''
    Performs best on evaluation data.

    :param data: evaluation data.
    :param index_: best model index, starts from 0.
    :return: eer and score_file.
    '''
    # Load best model.
    path_to_model = '../data/MODELS/best_model_' + data['dataset'] + str(
        index_) + '.h5'
    best_model = keras.models.load_model(path_to_model)
    # Predict on best model.
    eval_predictions = best_model.predict(data['x_data'], verbose=0)
    eva_eer, _, _, eval_score_file = scoring(probabilities=eval_predictions,
                                             labels=data['labels'],
                                             file_list=data['file_list'],
                                             indexes=data['indexes'])
    save(eval_score_file, data['dataset'] + str(index_) + 'eval.txt')

    return {
        'eval_eer': eva_eer,
        'eval_score_file': eval_score_file,
    }


# Feature list for loop.
feature_list = ['cqcc']
evaluate = False
for feature_ in feature_list:
    # Set load_eval_data to True when new feature type selected.
    load_eval_data = False
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
        network = {
            'n_dense': len(mlp_params['dense_units'][line].split('-')),
            'dense_units': int(mlp_params['dense_units'][line].split('-')[0]),
            'activation': mlp_params['activation'][line],
            'dropout': Dropout,
            'dropout_rate': 0.4,
            'optimizer': 'sgd'
        }
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
        eval_data = []
        while (epoch < 10000) and (not done_looping):
            # Report '1' for first epoch, 'n_epochs' for last epoch.
            epoch += 1
            # Train model.
            metrics = keras_model.fit(x_train,
                                      y_train,
                                      validation_data=(x_test, y_test),
                                      epochs=1,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      verbose=0)
            predictions = keras_model.predict(x_test, verbose=0,
                                              batch_size=batch_size)
            this_eer, _, _, score_file = scoring(probabilities=predictions,
                                                 labels=test_labels,
                                                 file_list=test_file_list,
                                                 indexes=test_indexes)
            dev_eer.append(this_eer)
            val_acc.append(metrics.history['val_acc'][0] * 20)
            print('Epoch = {}, Eer (validation) = {}, accuracy(train) = {:,.4}'
                  .format(epoch, this_eer, metrics.history['acc'][0]))
            early_stop += 1
            if this_eer < best_eer * 0.995:
                # If best eer is found.
                best_eer = this_eer
                save(score_file, feature_ + str(line) + 'dev.txt')
                # Save best model.
                model_file = f'../data/MODELS/best_model_{feature_}{str(line)}.h5'
                keras_model.save(model_file)
                print(f'Best eer = {best_eer}, model saved.\n')
                early_stop = 0
            if early_stop == 300:
                done_looping = True
                break
        if load_eval_data:
            print('Loading evaluation data...')
            eval_data = read_data(feature_, 'eval')
            load_eval_data = False
        eval_eer = evaluate_model(eval_data, line)[
            'eval_eer'] if evaluate else 0
        save_excel('slt-mlp.xlsx', feature_, line)

# Evaluate model.
ltas_results = []
eval_data = read_data('mfcc', 'eval')
for index in range(1):
    evaluation = evaluate_model(eval_data, index)
    ltas_results.append(evaluation['eval_eer'])
