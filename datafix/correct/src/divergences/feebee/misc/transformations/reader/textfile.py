from absl import app
from absl import flags
from absl import logging
import json
import numpy as np
import os.path as path
import time

from . import generic

FLAGS = flags.FLAGS

flags.DEFINE_string("file_path", ".", "Path to the features and labels file")
flags.DEFINE_string("file_json", None, "JSON file containing features and labels")
flags.DEFINE_string("file_dict_features", None, "Property containing the features")
flags.DEFINE_string("file_dict_labels", None, "Property containing the labels")
flags.DEFINE_integer("file_startindex", 0, "Start index of the entries in the file")
flags.DEFINE_integer(
    "file_endindex", -1, "End index of the entries in the file (-1 means all)"
)
flags.DEFINE_integer("text_numsentences", 10, "Max number of sentences")
flags.DEFINE_string("text_decodeformat", "utf-8", "Decoding format")


def test_argument_and_file(folder_path, arg_name):
    if arg_name not in FLAGS.__flags.keys() or not FLAGS.__flags[arg_name].value:
        raise app.UsageError(
            "--{} is a required argument when runing the tool with matrices.".format(
                arg_name
            )
        )

    arg_val = FLAGS.__flags[arg_name].value

    if not path.exists(path.join(folder_path, arg_val)):
        raise app.UsageError(
            "File '{}' given by '--{}' does not exists in the specified folder '{}'.".format(
                arg_val, arg_name, folder_path
            )
        )


def load_and_log_json():
    logging.log(logging.DEBUG, "Start loading file '{}'".format(FLAGS.file_json))
    start = time.time()
    with open(path.join(FLAGS.file_path, FLAGS.file_json)) as f:
        content = f.readlines()
    if FLAGS.file_endindex < 0:
        content = content[FLAGS.file_startindex :]
    else:
        content = content[FLAGS.file_startindex : FLAGS.file_endindex]
    features = []
    labels = []
    samples = len(content)
    for line in content:
        review = json.loads(line)
        features.append(review[FLAGS.file_dict_features])
        labels.append(int(review[FLAGS.file_dict_labels]))
    end = time.time()
    logging.log(
        logging.DEBUG, "'{}' loaded in {} seconds".format(FLAGS.file_json, end - start)
    )

    logging.log(
        logging.INFO,
        "'{}' loaded as lists with '{}' samples".format(FLAGS.file_json, samples),
    )
    return features, labels, samples


def read(fn):
    if fn is None:
        raise app.UsageError(
            "reading from a textfile must provide a function (TF-Hub Module or Torchhub model"
        )

    test_argument_and_file(FLAGS.file_path, "file_json")

    features, labels, samples = load_and_log_json()

    indices = list(range(samples))
    num_batches = int(samples + (FLAGS.batch_size - 1)) // FLAGS.batch_size
    new_features = []

    cnt = 0
    for select in generic.chunks(indices, FLAGS.batch_size, samples):
        if cnt == 0 or cnt % 10 == 9 or cnt == num_batches - 1:
            logging.log(
                logging.DEBUG, "List -- Processing {}/{}".format(cnt + 1, num_batches)
            )

        text = [features[x] for x in select]

        res = fn(text)
        new_features.extend(res.reshape(res.shape[0], -1))

        cnt += 1

    features = np.array(new_features)
    samples = features.shape[0]
    dim = features.shape[1]

    return features, dim, samples, labels
