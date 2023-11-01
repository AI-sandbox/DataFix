from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
from sklearn.model_selection import train_test_split

import transformations.reader.generic as generic_reader
import transformations.tfhub_module as tfhub_module
import transformations.torchhub_model as torchhub_model
import transformations.pca as pca
import transformations.nca as nca
import transformations.random_proj as random_proj

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "variant",
    None,
    ["matrix", "textfile", "folder", "mnist_data", "cifar_data", "tfds", "torchvision"],
    "Input for running the tool",
)
flags.DEFINE_list(
    "transformations",
    [],
    "List of transformations (can be empty, which exports the raw features) to be applied in that order (starting from the second only using a matrix as a input",
)
flags.DEFINE_integer("subsamples", None, "Number of subsamples to export")

flags.DEFINE_string(
    "export_path",
    ".",
    "Path to folder (should exist) where the features and labels matrices should be stored",
)
flags.DEFINE_string("export_features", None, "Features export file name")
flags.DEFINE_string("export_labels", None, "Labels export file name")


def _get_transform_fns():
    fns = []
    for t in FLAGS.transformations:
        val = t.strip().lower()
        if val == "tfhub_module":
            fns.append(
                tfhub_module.load_and_apply if len(fns) == 0 else tfhub_module.apply
            )
        elif val == "pca":
            fns.append(pca.load_and_apply if len(fns) == 0 else pca.apply)
        elif val == "nca":
            fns.append(nca.load_and_apply if len(fns) == 0 else nca.apply)
        elif val == "random_proj":
            fns.append(
                random_proj.load_and_apply if len(fns) == 0 else random_proj.apply
            )
        elif val == "torchhub_model":
            fns.append(
                torchhub_model.load_and_apply if len(fns) == 0 else torchhub_model.apply
            )
        else:
            raise app.UsageError("Transformation '{}' is not valid!".format(t))

    return fns


def main(argv):
    if not path.exists(FLAGS.export_path):
        raise app.UsageError(
            "Path to the export folder '{}' needs to exist!".format(FLAGS.export_path)
        )

    if FLAGS.variant == "matrix" and len(FLAGS.transformations) == 0:
        raise app.UsageError(
            "Loading and rexporting the labels and features matrix without transformation is stupid! Use the command line and 'cp'!"
        )

    # Apply transformations
    transform_fns = _get_transform_fns()
    for i, fn in enumerate(transform_fns):
        if i == 0:
            features, dim, samples, labels = fn()
        else:
            features, dim, samples, labels = fn(features, dim, samples, labels)

    if len(transform_fns) == 0:
        features, dim, samples, labels = generic_reader.read()

    if (
        FLAGS.subsamples is not None
        and FLAGS.subsamples > 0
        and FLAGS.subsamples < samples
    ):
        logging.log(logging.INFO, "Subsampling {} sampels".format(FLAGS.subsamples))
        features, _, labels, _ = train_test_split(
            features,
            labels,
            test_size=None,
            train_size=FLAGS.subsamples,
            stratify=labels,
        )
        samples = FLAGS.subsamples

    # Export data
    export_folder = FLAGS.export_path
    features_path = path.join(export_folder, FLAGS.export_features)
    logging.log(
        logging.INFO,
        "Saving features with shape {} to '{}'".format(
            np.shape(features), features_path
        ),
    )
    np.save(features_path, features)

    labels_path = path.join(export_folder, FLAGS.export_labels)
    logging.log(
        logging.INFO,
        "Saving labels with shape {} to '{}'".format(np.shape(labels), labels_path),
    )
    np.save(labels_path, labels)


if __name__ == "__main__":
    flags.mark_flag_as_required("variant")
    flags.mark_flag_as_required("export_features")
    flags.mark_flag_as_required("export_labels")
    app.run(main)
