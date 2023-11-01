from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "export_path",
    ".",
    "Path to folder (should exist) where the features and labels matrices should be stored",
)
flags.DEFINE_string("export_features", None, "Features export file name")
flags.DEFINE_string("export_labels", None, "Labels export file name")

flags.DEFINE_integer("dim1", 2, "Number of tiles in the first dimension")
flags.DEFINE_integer("dim2", 2, "Number of tiles in the second dimension")
flags.DEFINE_integer("samples", 5000, "Number of samples per tile")
flags.DEFINE_float(
    "scale",
    0.125,
    "Scale for the normal distribtion centered in the middle of the tile.",
)


def main(argv):
    if not path.exists(FLAGS.export_path):
        raise app.UsageError(
            "Path to the export folder '{}' needs to exist!".format(FLAGS.export_path)
        )

    dim1 = FLAGS.dim1  # x-dimension
    dim2 = FLAGS.dim2  # y-dimension
    n_bucket = FLAGS.samples
    random_scale = FLAGS.scale

    data = {"x1": [], "x2": [], "label": []}
    for xcoor in range(dim1):
        for ycoor in range(dim2):
            if xcoor % 2 == 0 and ycoor % 2 == 0:
                label = 1
            elif xcoor % 2 == 1 and ycoor % 2 == 1:
                label = 1
            else:
                label = 0
            data["x1"].extend(np.random.normal(xcoor + 0.5, random_scale, n_bucket))
            data["x2"].extend(np.random.normal(ycoor + 0.5, random_scale, n_bucket))
            data["label"].extend([label] * n_bucket)

    features = np.column_stack([data["x1"], data["x2"]])
    labels = np.array(data["label"])

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
    flags.mark_flag_as_required("export_features")
    flags.mark_flag_as_required("export_labels")
    app.run(main)
