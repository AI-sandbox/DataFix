from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math

FLAGS = flags.FLAGS

flags.DEFINE_string("tfhub_module", None, "TF Hub module handle")
flags.DEFINE_string("tfhub_tag", None, "TF Hub tag to use")
flags.DEFINE_enum("tfhub_type", "image", ["image", "text"], "Type of the tf hub module")
flags.DEFINE_boolean("tfhub_newmodule", True, "New or old module type (call type)")

from .reader import generic
from .reader.tfds import get_batch_iterator
from .reader.folder import read as read_from_folder
from .reader.tfds import read as read_from_tfds
from .reader.textfile import read as read_from_textfile
from .reader.matrix import apply_fn_matrices


def setup():
    if not FLAGS.tfhub_module:
        raise app.UsageError("--tfhub_module has to be specified!")


def get_text_fn(to_numpy=False, string_input=False):
    embedding = FLAGS.tfhub_module
    if FLAGS.tfhub_newmodule:
        module = hub.KerasLayer(
            embedding, input_shape=[], dtype=tf.string, trainable=False
        )

    else:
        module = hub.load(FLAGS.tfhub_module)

    if to_numpy:
        output = lambda z: module(z).numpy()
    else:
        output = lambda z: module(z)

    if string_input and FLAGS.tfhub_newmodule:
        compute = lambda z: output(tf.constant(z))
    else:
        compute = lambda z: output(z)

    return compute


def load_and_apply_from_tfds():
    if FLAGS.tfhub_type == "image":
        if FLAGS.tfhub_tag is None:
            module = hub.load(FLAGS.tfhub_module)
        elif FLAGS.tfhub_tag == "":
            module = hub.load(FLAGS.tfhub_module, tags=[])
        else:
            module = hub.load(FLAGS.tfhub_module, tags=[FLAGS.tfhub_tag])

        compute = lambda z: module.signatures["default"](z)["default"]
        return read_from_tfds(compute, True, True, True)

    # ELSE Text
    return read_from_tfds(get_text_fn(False, False), True, False, False)


def load_and_apply_from_folder():
    if FLAGS.tfhub_type == "text":
        raise app.UsageError(
            "Hub module for Text only alowed in combination with tfds or textfile so far!"
        )

    if FLAGS.tfhub_tag is None:
        module = hub.load(FLAGS.tfhub_module)
    elif FLAGS.tfhub_tag == "":
        module = hub.load(FLAGS.tfhub_module, tags=[])
    else:
        module = hub.load(FLAGS.tfhub_module, tags=[FLAGS.tfhub_tag])

    def apply_tf_fn(images):
        images = tf.cast(images, tf.float32)
        compute = lambda z: module.signatures["default"](z)["default"]
        return compute(images)

    return read_from_folder(apply_tf_fn)


def load_and_apply_from_textfile():
    if FLAGS.tfhub_type == "image":
        raise app.UsageError("Hub module for images not allowed for text files inputs!")

    return read_from_textfile(get_text_fn(True, True))


def load_and_apply():
    setup()

    if FLAGS.variant == "tfds":
        return load_and_apply_from_tfds()

    if FLAGS.variant == "folder":
        return load_and_apply_from_folder()

    if FLAGS.variant == "textfile":
        return load_and_apply_from_textfile()

    # if not directly supported -> use generic reader
    features, dim, samples, labels = generic.read()
    return apply(features, dim, samples, labels)


def apply(features, dim, samples, labels):
    setup()

    if FLAGS.tfhub_type == "text":
        raise app.UsageError(
            "Hub module for Text only alowed in combination with tfds or textfile so far!"
        )

    if FLAGS.tfhub_tag is None:
        module = hub.load(FLAGS.tfhub_module)
    elif FLAGS.tfhub_tag == "":
        module = hub.load(FLAGS.tfhub_module, tags=[])
    else:
        module = hub.load(FLAGS.tfhub_module, tags=[FLAGS.tfhub_tag])

    def fn(images):
        # TODO dynamically check input signature
        images = [
            tf.image.resize(
                tf.cast(x, tf.float32), [FLAGS.resize_height, FLAGS.resize_width]
            )
            for x in images
        ]
        images = tf.stack(images)
        compute = lambda z: module.signatures["default"](z)["default"]
        return compute(images).numpy()

    return apply_fn_matrices(features, dim, samples, labels, fn)
