from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import os.path as path
import random
import time
import tensorflow as tf

from . import generic

FLAGS = flags.FLAGS

flags.DEFINE_string("folder_path", ".", "Path the folder having images per subfolder")
flags.DEFINE_string(
    "folder_mapping_path", None, "Path the subfolder name to class mapping"
)
flags.DEFINE_integer(
    "folder_samples_per_class",
    None,
    "Only take a certain number of samples per class",
    lower_bound=1,
)


def get_paths_dict(folder_path):
    if not FLAGS.folder_mapping_path:
        raise NotImplementedError(
            "Dynamic mapping folders to classes not yet implemented! Define the argument --folder_mapping_path !"
        )

    if not path.exists(FLAGS.folder_mapping_path):
        raise app.UsageError("--folder_mapping_path needs to exist!")

    mapping = {}
    cnt = 1
    with open(FLAGS.folder_mapping_path, "r") as f:
        line = f.readline()
        while line:
            s = line.strip()
            if s == "":
                continue
            mapping[s] = cnt
            cnt += 1
            line = f.readline()

    files_per_label = {}
    for r, d, fs in os.walk(folder_path):
        cat = r.rsplit("/", 1)[1]
        if cat.startswith("n"):
            label = mapping[cat]
            files_per_label[label] = list(map(lambda z: os.path.join(r, z), fs))

    return files_per_label


def test_argument_and_path(arg_name):
    if arg_name not in FLAGS.__flags.keys() or not FLAGS.__flags[arg_name].value:
        raise app.UsageError(
            "--{} is a required argument when runing the tool with 'folder'.".format(
                arg_name
            )
        )

    arg_val = FLAGS.__flags[arg_name].value

    if not path.exists(arg_val):
        raise app.UsageError(
            "Path '{}' given by '--{}' does not exists.".format(arg_val, arg_name)
        )


def read_img(path):
    image_size = [FLAGS.resize_height, FLAGS.resize_width]
    img_raw = tf.io.read_file(path)
    images = tf.image.decode_jpeg(img_raw, channels=FLAGS.input_channels)
    # TODO repeate to 3 channels if needed similiar to np.repeat(x[...,None],3,axis=2)
    return tf.image.resize(images, image_size) / 255.0


def read_paths_to_matrix(paths):
    features = None
    pos = 0
    num_samples = len(paths)

    batch_size = FLAGS.batch_size
    num_batches = (len(paths) + batch_size - 1) // batch_size
    cnt = 0
    for select in generic.chunks(paths, batch_size, len(paths)):
        if cnt == 0 or cnt % 10 == 9 or cnt == num_batches - 1:
            print(
                "Get samples from folder -- Iteration: {} out of {}".format(
                    cnt + 1, num_batches
                )
            )
        cnt += 1

        images = tf.stack(list(map(lambda z: read_img(z), select)))
        x = images.numpy()
        x = x.reshape(x.shape[0], -1)
        if features is None:
            features = np.zeros((num_samples, x.shape[1]), dtype=float)
        features[pos : (pos + x.shape[0]), :] = x
        pos += x.shape[0]

    return features


def read(apply_fn=None, tf_fn=True, use_output=True):
    test_argument_and_path("folder_path")

    files_per_label = get_paths_dict(FLAGS.folder_path)

    batch_size = FLAGS.batch_size

    if FLAGS.folder_samples_per_class:
        num_samples = sum(
            [
                FLAGS.folder_samples_per_class
                if FLAGS.folder_samples_per_class < len(v)
                else len(v)
                for (k, v) in files_per_label.items()
            ]
        )
    else:
        num_samples = sum([len(v) for (k, v) in files_per_label.items()])

    features = None
    y = []
    pos = 0

    for i, (k, v) in enumerate(files_per_label.items()):
        paths = np.array(v)
        indices = list(range(len(v)))

        if FLAGS.folder_samples_per_class and FLAGS.folder_samples_per_class < len(v):
            random.shuffle(indices)
            num_batches = (
                FLAGS.folder_samples_per_class + batch_size - 1
            ) // batch_size
            num_this_class = FLAGS.folder_samples_per_class
        else:
            num_batches = (len(v) + batch_size - 1) // batch_size
            num_this_class = len(v)
        cnt = 0
        for select in generic.chunks(indices, batch_size, num_this_class):
            if cnt == 0 or cnt % 10 == 9 or cnt == num_batches - 1:
                print(
                    "Get samples from folder -- Iteration: {} out of {} for class {} out of {}".format(
                        cnt + 1, num_batches, i + 1, len(files_per_label.keys())
                    )
                )
            cnt += 1

            images = tf.stack(list(map(lambda z: read_img(z), paths[select])))
            if apply_fn is None:
                x = images.numpy()
            else:
                if use_output:
                    x = apply_fn(images).numpy() if tf_fn else apply_fn(images.numpy())
                elif tf_fn:
                    apply_fn(images).numpy()
                else:
                    apply_fn(images.numpy())

            y.extend([k] * len(select))
            if use_output:
                x = x.reshape(x.shape[0], -1)
                if features is None:
                    features = np.zeros((num_samples, x.shape[1]), dtype=float)
                features[pos : (pos + x.shape[0]), :] = x
                pos += x.shape[0]

    labels = np.array(y)

    if not use_output:
        return labels

    samples = np.shape(features)[0]
    dim = np.shape(features)[1]

    if len(labels.shape) != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if labels.shape[0] != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features, dim, samples, labels
