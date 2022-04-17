# -*- coding: utf-8 -*-
import numpy
from keras.backend import clear_session
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

import source.scoring as scoring
from source.features import load_features
from source.scoring import create


def create_network(input_shape):
    # Clear gpu memory first.
    clear_session()

    # Set input layer.
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu",
                     input_shape=input_shape))

    # Set intermediate layer(s).
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    # Third layer.
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    # Set output layer.
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adadelta",
                  metrics=["accuracy"])

    # Print out model summary.
    model.summary()

    return model


# Definitions
path_to_model = "../data/models/"

# Load training data, development data and labels.
feature = 0
feature_list = ["ltas", "power_spectrum"]
x_train, y_train, _, _, batch_size, _ = load_features(feature_list[feature],
                                                      "train")
x_test, y_test, test_labels, indexes, _, test_file_list = load_features(
    feature_list[feature], "dev")

# Convert data to 2D.
img_rows = 51
img_cols = 10
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Create network.
print("Creating network...")
# Create keras model.
keras_model = create_network(input_shape=(img_rows, img_cols, 1))

# Create placeholders to store metrics.
dev_eer = []
best_eer = numpy.inf

# Early stopping parameters.
early_stop = 0
epoch = 0
done_looping = False
while (epoch < 10000) and (not done_looping):
    metrics = keras_model.fit(x_train, y_train, epochs=1,
                              batch_size=batch_size, shuffle=True, verbose=0)
    predictions = keras_model.predict(x_test, verbose=0, batch_size=batch_size)
    score_file = create(predictions, labels=test_labels,
                        file_list=test_file_list, indexes=None)
    this_eer = scoring.calculate_eer(score_file)
    dev_eer.append(this_eer)
    print("Epoch={}, val_eer={}, accuracy(train) = {:,.4}".format(
        epoch, this_eer, metrics.history["acc"][0]))

    if this_eer < best_eer * 0.995:
        # If best eer is found.
        best_eer = this_eer
        scoring.save(score_file, feature_list[feature] + "dev_cnn.txt")

        # Save best model.
        model_filename = feature_list[feature] + "_cnn.h5"
        keras_model.save(path_to_model + model_filename)
        print("Best eer = {}, model saved.\n".format(best_eer))
        early_stop = 0

    if early_stop == 300:
        done_looping = True
        break

    early_stop += 1
    epoch += 1

# Save process metrics here.
