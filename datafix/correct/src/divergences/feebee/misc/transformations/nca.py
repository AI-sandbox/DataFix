from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path

# import torch
# from torchnca import NCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from .reader import generic

FLAGS = flags.FLAGS

flags.DEFINE_integer("nca_dim", None, "Remaining ", lower_bound=0)
flags.DEFINE_string(
    "nca_fit_features",
    None,
    "Path to different matrix to fit PCA to. If 'None', the used input features will be selected",
)
flags.DEFINE_string(
    "nca_fit_labels",
    None,
    "Path to different matrix to fit PCA to. If 'None', the used input labels will be selected",
)
# flags.DEFINE_integer("nca_batch_size", None, "NCA Batch Size")
# flags.DEFINE_integer("nca_weight_decay", 10, "NCA Batch Size")
# flags.DEFINE_float("nca_lr", 1e-4, "NCA Batch Size")
# flags.DEFINE_boolean("nca_normalize", True, "NCA Batch Size")


def load_and_apply():
    # use generic reader
    features, dim, samples, labels = generic.read()
    return apply(features, dim, samples, labels)


def apply(features, dim, samples, labels, nca_fit_features=None, nca_fit_labels=None):
    if not FLAGS.nca_dim or not (FLAGS.nca_dim > 0 and FLAGS.nca_dim < dim):
        raise app.UsageError(
            "--nca_dim needs to be specified, strictly larger than 0 and lower than the dimension of your input!"
        )

    nca = NeighborhoodComponentsAnalysis(FLAGS.nca_dim)
    if FLAGS.nca_fit_features:
        if not path.exists(FLAGS.nca_fit_features):
            raise app.UsageError(
                "File '{}' given by '--nca_fit_features' does not exists.".format(
                    FLAGS.nca_fit_features
                )
            )
        nca_fit_features = np.load(FLAGS.nca_fit_features)
    if FLAGS.nca_fit_labels:
        if not path.exists(FLAGS.nca_fit_labels):
            raise app.UsageError(
                "File '{}' given by '--nca_fit_labels' does not exists.".format(
                    FLAGS.nca_fit_labels
                )
            )
        nca_fit_labels = np.load(FLAGS.nca_fit_labels)

    if nca_fit_labels is not None and nca_fit_features is not None:
        nca.fit(nca_fit_features, nca_fit_labels)
    else:
        nca.fit(features, labels)

    features = nca.transform(features)

    return features, FLAGS.nca_dim, samples, labels

    """
    nca = NCA(dim=FLAGS.nca_dim, init="identity")
    if FLAGS.nca_fit_features:
        if not path.exists(FLAGS.nca_fit_features):
            raise app.UsageError("File '{}' given by '--nca_fit_features' does not exists.".format(FLAGS.nca_fit_features))
        nca_fit_features = np.load(FLAGS.nca_fit_features)
    if FLAGS.nca_fit_labels:
        if not path.exists(FLAGS.nca_fit_label):
            raise app.UsageError("File '{}' given by '--nca_fit_labels' does not exists.".format(FLAGS.nca_fit_labels))
        nca_fit_labels = np.load(FLAGS.nca_fit_labels)

    if nca_fit_labels is not None and nca_fit_features is not None:
        X = torch.from_numpy(nca_fit_features).float()
        y = torch.from_numpy(nca_fit_labels).long()
    else:
        X = torch.from_numpy(features).float()
        y = torch.from_numpy(np.array(labels)).long()

    if torch.cuda.is_available():
        X = X.to('cuda')
        y = y.to('cuda')

    print(X)
    print(y)

    nca.train(X, y, batch_size=FLAGS.nca_batch_size, weight_decay=FLAGS.nca_weight_decay, lr=FLAGS.nca_lr, normalize=FLAGS.nca_normalize)

    X = torch.from_numpy(features)
    if torch.cuda.is_available():
        X = X.to('cuda')
    features = nca(X).numpy()

    return features, FLAGS.nca_dim, samples, labels
    """
