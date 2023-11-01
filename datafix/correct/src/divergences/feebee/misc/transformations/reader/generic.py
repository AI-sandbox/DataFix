from absl import flags

from . import matrix
from . import folder
from . import mnist_data
from . import cifar_data
from . import tfds

FLAGS = flags.FLAGS

flags.DEFINE_integer("resize_height", 299, "Resize image new height", lower_bound=0)
flags.DEFINE_integer("resize_width", 299, "Resize image new width", lower_bound=0)
flags.DEFINE_integer(
    "batch_size", 64, "Batch size to loop through the samples", lower_bound=0
)


def chunks(l, n, cnt):
    """Yield successive n-sized chunks from l first cnt elements."""
    for i in range(0, cnt, n):
        if i + n > cnt:
            yield l[i:cnt]
        else:
            yield l[i : i + n]


def read():
    # Read data
    if FLAGS.variant == "matrix":
        features, dim, samples, labels = matrix.read()
    elif FLAGS.variant == "folder":
        features, dim, samples, labels = folder.read()
    elif FLAGS.variant == "mnist_data":
        features, dim, samples, labels = mnist_data.read()
    elif FLAGS.variant == "cifar_data":
        features, dim, samples, labels = cifar_data.read()
    elif FLAGS.variant == "tfds":
        features, dim, samples, labels = tfds.read()
    else:
        raise NotImplementedError(
            "Variant '{}' not yet implemented generically!".format(FLAGS.variant)
        )

    return features, dim, samples, labels
