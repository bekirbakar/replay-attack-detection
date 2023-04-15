# -*- coding: utf-8 -*-
import joblib
import keras
import numpy
import pandas as pd
import scipy.io
from keras.backend import clear_session


def deep_features(feature_list, dataset_index, subset_list, file_list, path_to_excel, filename):
    start, indexes, end = 0, 0, 0
    feat_dir = '../data/features/'
    for feature in feature_list:
        # Read parameters from excel file (.xlsx).
        xl = pd.ExcelFile(path_to_excel)
        df = xl.parse('mlp-parameters')
        dataset = df['dataset'][dataset_index]
        path = '/' + dataset.upper() + '/' + dataset

        # Load model.
        model = keras.models.load_model(filename)
        model.summary()

        # Get last hidden layer of deep NN.
        intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer('hidden3').output)

        for subset in subset_list:
            mat_file = feat_dir + dataset.upper() + '/mat_files/' + subset

            # Load data.
            if feature == 'ltas':
                data = joblib.load(feat_dir + path + '_' + subset)[0]
            else:
                data = joblib.load(feat_dir + path + '_' + subset)
                data, indexes = data[0], data[2]
                start = 0
                end = indexes[0]

            # Convert to numpy array.
            data = numpy.asarray(data)

            # Convert data types to float32.
            data = data.astype('float32')
            # Loop through files.
            for file in range(len(file_list)):
                # Extract features from intermediate layer.
                buffer = data[file:(file + 1)] if feature == 'ltas' else data[start:end]
                intermediate_output = intermediate_layer_model.predict(buffer)

                # Save feature per file.
                mdict = {'data': intermediate_output}
                scipy.io.savemat(mat_file + file_list[file] + '.mat', mdict, appendmat=True, format='5',
                                 long_field_names=False, do_compression=False, oned_as='row')
                if feature != 'ltas':
                    start = end
                    end = indexes[file] + start

    clear_session()
