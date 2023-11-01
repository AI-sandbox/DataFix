from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .timed_logger import TimedLogger

FLAGS = flags.FLAGS

flags.DEFINE_list("l2_regs", [0.0], "L2 regularization (list) of the last layer")
flags.DEFINE_integer("epochs", 100, "Number of epochs to train")
flags.DEFINE_list("sgd_lrs", [0.1], "SGD learning rate (list)")
flags.DEFINE_float("sgd_momentum", 0.9, "SGD momentum")

KEY_PATTERN = "l2={0}, lr={1}"


def train_model_cross_entropy(
    features_train,
    labels_train,
    features_test,
    labels_test,
    classes,
    dimension,
    l2_reg,
    sgd_lr,
):
    with TimedLogger(
        "Training the linear layer with SGD(lr={}, momentum={}), batch_size={}, L2_Reg={}, epochs={}".format(
            sgd_lr, FLAGS.sgd_momentum, FLAGS.batch_size, l2_reg, FLAGS.epochs
        )
    ):
        model = keras.models.Sequential(
            [
                keras.layers.Dense(
                    classes,
                    input_shape=(dimension,),
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                )
            ]
        )
        model.compile(
            optimizer=keras.optimizers.SGD(
                learning_rate=sgd_lr, momentum=FLAGS.sgd_momentum
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            features_train,
            labels_train,
            epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            validation_data=(features_test, labels_test),
            verbose=0,
        )

    loss_accuracy = model.evaluate(features_test, labels_test, verbose=0)

    logging.log(
        logging.INFO, "Loss and accuracy cross_entropy: {0}".format(loss_accuracy)
    )

    return loss_accuracy[1]


def eval_from_matrices(train_features, test_features, train_labels, test_labels):
    # Adjust labels
    unique_classes = list(np.unique(np.concatenate((train_labels, test_labels))))
    for i in range(len(train_labels)):
        train_labels[i] = unique_classes.index(train_labels[i])
    for i in range(len(test_labels)):
        test_labels[i] = unique_classes.index(test_labels[i])

    with TimedLogger("Normalizing features using MinMaxScaler"):
        minmax = MinMaxScaler(feature_range=(-1, 1), copy=True)
        train_features = minmax.fit_transform(train_features)
        test_features = minmax.transform(test_features)

    classes = len(unique_classes)
    dim = train_features.shape[1]

    results = {}

    for l2_reg in sorted([float(x) for x in FLAGS.l2_regs]):
        for sgd_lr in sorted([float(x) for x in FLAGS.sgd_lrs]):
            acc = train_model_cross_entropy(
                train_features,
                train_labels,
                test_features,
                test_labels,
                classes,
                dim,
                l2_reg,
                sgd_lr,
            )
            results[KEY_PATTERN.format(l2_reg, sgd_lr)] = 1.0 - acc

    return results
