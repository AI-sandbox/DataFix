from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
from sklearn.random_projection import GaussianRandomProjection

from .reader import generic

FLAGS = flags.FLAGS

flags.DEFINE_integer("random_proj_dim", None, "Remaining dimensions", lower_bound=0)
flags.DEFINE_integer("random_proj_seed", None, "Seed for dim reduction")


def load_and_apply():
    # use generic reader
    features, dim, samples, labels = generic.read()
    return apply(features, dim, samples, labels)


def apply(features, dim, samples, labels, pca_fit_matrix=None):
    if not FLAGS.random_proj_dim or not (
        FLAGS.random_proj_dim > 0 and FLAGS.random_proj_dim < dim
    ):
        raise app.UsageError(
            "--random_proj_dim needs to be specified, strictly larger than 0 and lower than the dimension of your input!"
        )

    if FLAGS.random_proj_seed is None:
        transformer = GaussianRandomProjection(n_components=FLAGS.random_proj_dim)
    else:
        rng = np.random.RandomState(FLAGS.random_proj_seed)
        transformer = GaussianRandomProjection(
            n_components=FLAGS.random_proj_dim, random_state=rng
        )

    features = transformer.fit_transform(features)

    return features, FLAGS.random_proj_dim, samples, labels
