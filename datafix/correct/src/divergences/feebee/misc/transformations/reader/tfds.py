from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("tfds_name", None, "Dataset name")
flags.DEFINE_string("tfds_split", None, "Dataset split")
flags.DEFINE_string("tfds_feature", "image", "Feature name")
flags.DEFINE_string("tfds_label", "label", "Label name")
flags.DEFINE_integer("tfds_subset", 0, "Take a subset (0 for all)", lower_bound=0)
flags.DEFINE_string("tfds_datadir", None, "Different data dir")


def get_batch_iterator(devide=False, resize=False):
    if not FLAGS.tfds_name:
        raise app.UsageError("--tfds_name has to be specified!")
    if not FLAGS.tfds_split:
        raise app.UsageError("--tfds_split has to be specified!")

    if FLAGS.tfds_datadir is not None:
        dataset, info = tfds.load(
            FLAGS.tfds_name,
            split=FLAGS.tfds_split,
            with_info=True,
            data_dir=FLAGS.tfds_datadir,
        )
    else:
        dataset, info = tfds.load(
            FLAGS.tfds_name, split=FLAGS.tfds_split, with_info=True
        )

    if FLAGS.tfds_subset < 1:
        num_batches = int(
            (info.splits[FLAGS.tfds_split].num_examples + (FLAGS.batch_size - 1))
            // FLAGS.batch_size
        )
    else:
        num_batches = int(
            (
                min(info.splits[FLAGS.tfds_split].num_examples, FLAGS.tfds_subset)
                + (FLAGS.batch_size - 1)
            )
            // FLAGS.batch_size
        )

    def read_sample(x):
        if devide:
            features = tf.cast(x[FLAGS.tfds_feature], tf.float32) / 255.0
        else:
            features = x[FLAGS.tfds_feature]
        if resize:
            # TODO first crop than downsize
            features = tf.image.resize(
                features, [FLAGS.resize_height, FLAGS.resize_width]
            )
        return features, x[FLAGS.tfds_label]

    if FLAGS.tfds_subset > 0:
        dataset = dataset.take(FLAGS.tfds_subset)

    return dataset.map(read_sample).batch(FLAGS.batch_size), num_batches


def read(apply_fn=None, tf_fn=True, devide=True, resize=True):
    iterator, num_batches = get_batch_iterator(devide, resize)
    features = []
    y = []

    cnt = 0
    for objects, labels in iterator:
        if cnt == 0 or cnt % 10 == 9 or cnt == num_batches - 1:
            logging.log(
                logging.INFO,
                "Get samples from TFDS -- Processing {}/{}".format(
                    cnt + 1, num_batches
                ),
            )

        if apply_fn is None:
            x = objects.numpy()
        else:
            x = apply_fn(objects).numpy() if tf_fn else apply_fn(objects.numpy())
        features.extend(x.reshape(x.shape[0], -1))

        y.extend(labels.numpy())

        cnt += 1

    labels = np.array(y)
    features = np.array(features)
    samples = features.shape[0]
    dim = features.shape[1]

    if len(labels.shape) != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if labels.shape[0] != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features, dim, samples, labels
