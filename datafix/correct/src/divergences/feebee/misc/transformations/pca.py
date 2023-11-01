from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
from sklearn.decomposition import PCA

from .reader import generic

FLAGS = flags.FLAGS

flags.DEFINE_integer("pca_dim", None, "Remaining ", lower_bound=0)
flags.DEFINE_string(
    "pca_fit_matrix_path",
    None,
    "Path to different matrix to fit PCA to. If 'None', the used input matrix will be selected",
)


def load_and_apply():
    # use generic reader
    features, dim, samples, labels = generic.read()
    return apply(features, dim, samples, labels)


def apply(features, dim, samples, labels, pca_fit_matrix=None):
    if not FLAGS.pca_dim or not (FLAGS.pca_dim > 0 and FLAGS.pca_dim < dim):
        raise app.UsageError(
            "--pca_dim needs to be specified, strictly larger than 0 and lower than the dimension of your input!"
        )

    pca = PCA(FLAGS.pca_dim)
    if FLAGS.pca_fit_matrix_path:
        if not path.exists(FLAGS.pca_fit_matrix_path):
            raise app.UsageError(
                "File '{}' given by '--pca_fit_matrix_path' does not exists.".format(
                    FLAGS.pca_fit_matrix_path
                )
            )
        pca_fit_matrix = np.load(FLAGS.pca_fit_matrix_path)

    if pca_fit_matrix is not None:
        pca.fit(pca_fit_matrix)
    else:
        pca.fit(features)

    features = pca.transform(features)

    return features, FLAGS.pca_dim, samples, labels
