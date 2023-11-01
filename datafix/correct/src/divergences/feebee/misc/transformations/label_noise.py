from absl import app
from absl import flags
from absl import logging
import numpy as np

from .reader import generic

FLAGS = flags.FLAGS

flags.DEFINE_float(
    "label_noise_amount",
    None,
    "Percentage of random label assignment between 0.0 and 1.0",
    lower_bound=0.0,
    upper_bound=1.0,
)


def random_flip(labels, samples, fraction, copy=False):
    label_values = np.unique(labels)
    idx = np.random.choice(samples, size=int(fraction * samples), replace=False)

    if not copy:
        labels[idx] = np.random.choice(label_values, size=len(idx))
        return labels

    labels_new = np.copy(labels)
    labels_new[idx] = np.random.choice(label_values, size=len(idx))
    return labels_new


def load_and_apply():
    # use generic reader
    features, dim, samples, labels = generic.read()
    return apply(features, dim, samples, labels)


def apply(features, dim, samples, labels):
    if not FLAGS.label_noise_amount or FLAGS.label_noise_amount == 0.0:
        raise app.UsageError(
            "--label_noise_amount needs to be specified and strictly larger than 0!"
        )

    labels = random_flip(labels, samples, FLAGS.label_noise_amount, False)

    return features, dim, samples, labels
