from absl import app
from absl import flags
from absl import logging
import numpy as np
import math
import torch
from torchvision import transforms

FLAGS = flags.FLAGS

flags.DEFINE_string("torchhub_github", None, "Torch Hub github name")
flags.DEFINE_string("torchhub_name", None, "Torch Hub model name")
flags.DEFINE_string("torchhub_args", None, "Torch Hub model name")
flags.DEFINE_enum(
    "torchhub_adjust", "resize", ["resize", "pad", "mix"], "Variant to fit the input"
)
flags.DEFINE_integer(
    "torchhub_mininput", 224, "Minimum expected input by the torch hub module"
)
flags.DEFINE_string("torchhub_tokenizer_github", None, "Torch Hub github tokenizer")
flags.DEFINE_string("torchhub_tokenizer_name", None, "Torch Hub tokenizer name")
flags.DEFINE_string("torchhub_tokenizer_args", None, "Torch Hub tokenizer args")
flags.DEFINE_enum(
    "torchhub_type", "image", ["image", "text"], "Type of the tf hub module"
)
flags.DEFINE_integer("torchhub_maxinputlength", 512, "Maximum input token length")
flags.DEFINE_boolean(
    "torchhub_tokenoutput",
    False,
    "Output the embeddings per token instead of full sentences",
)
flags.DEFINE_integer(
    "torchhub_maxtokenoutput", 128, "Max number of tokens for the token output"
)

from .reader import generic
from .reader.folder import read as read_from_folder
from .reader.tfds import read as read_from_tfds
from .reader.textfile import read as read_from_textfile
from .reader.matrix import apply_fn_matrices


def setup():
    if not FLAGS.torchhub_github:
        raise app.UsageError("--torchhub_github has to be specified!")
    if not FLAGS.torchhub_name:
        raise app.UsageError("--torchhub_name has to be specified!")


def load_and_apply_from_folder():
    if FLAGS.torchhub_type == "text":
        raise app.UsageError(
            "Torch module for Text only alowed in combination with tfds or textfile so far!"
        )

    if FLAGS.torchhub_args is None:
        model = torch.hub.load(
            FLAGS.torchhub_github, FLAGS.torchhub_name, pretrained=True
        )
    else:
        model = torch.hub.load(
            FLAGS.torchhub_github,
            FLAGS.torchhub_name,
            FLAGS.torchhub_args,
            pretrained=True,
        )
    model.eval()

    new_features = []

    def get_features():
        def hook(model, input, output):
            res = output.cpu().detach().numpy()
            new_features.extend(res.reshape(res.shape[0], -1))

        return hook

    if FLAGS.torchhub_name == "alexnet" or FLAGS.torchhub_name == "vgg19":
        model.classifier[5].register_forward_hook(get_features())
    elif FLAGS.torchhub_name == "googlenet":
        model.dropout.register_forward_hook(get_features())
    else:
        raise NotImplementedError(
            "Hook for exported representation for model '{}' not implemented yet!".format(
                FLAGS.torchhub_name
            )
        )

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model.to("cuda")

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def fn(images):
        if images.dtype is not np.dtype(np.float32):
            images = images.astype(np.float32)
        images = [preprocess(x) for x in images]
        images = torch.stack(images)
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            images = images.to("cuda")

        with torch.no_grad():
            output = model(images)

    labels = read_from_folder(fn, False, False)

    features = np.array(new_features)
    samples = features.shape[0]
    dim = features.shape[1]

    if len(labels.shape) != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if labels.shape[0] != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features, dim, samples, labels


def get_text_fn(string_input=False):
    if FLAGS.torchhub_args is None:
        model = torch.hub.load(FLAGS.torchhub_github, FLAGS.torchhub_name)
    else:
        model = torch.hub.load(
            FLAGS.torchhub_github, FLAGS.torchhub_name, FLAGS.torchhub_args
        )
    if FLAGS.torchhub_args is None:
        tokenizer = torch.hub.load(
            FLAGS.torchhub_tokenizer_github, FLAGS.torchhub_tokenizer_name
        )
    else:
        tokenizer = torch.hub.load(
            FLAGS.torchhub_tokenizer_github,
            FLAGS.torchhub_tokenizer_name,
            FLAGS.torchhub_tokenizer_args,
        )

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model.to("cuda")

    def compute(input_string):
        input_ids = torch.tensor(
            [tokenizer.encode(input_string, add_special_tokens=True)]
        )
        if (
            FLAGS.torchhub_maxinputlength is not None
            and input_ids.shape[1] > FLAGS.torchhub_maxinputlength
        ):
            input_ids = input_ids[:, : FLAGS.torchhub_maxinputlength]
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        with torch.no_grad():
            model_res = model(input_ids)
            if FLAGS.torchhub_tokenoutput:
                # use the per token representation, concat and pad with 0's
                last_dim = model_res[0].shape[-1]
                num_tokens = model_res[0].shape[1]

                res = np.zeros((1, FLAGS.torchhub_maxtokenoutput * last_dim))
                if FLAGS.torchhub_maxtokenoutput < num_tokens:
                    res[:, :] = (
                        model_res[0]
                        .cpu()
                        .detach()
                        .numpy()[:, : FLAGS.torchhub_maxtokenoutput, :]
                        .reshape(-1, FLAGS.torchhub_maxtokenoutput * last_dim)
                    )
                else:
                    res[:, : (num_tokens * last_dim)] = (
                        model_res[0]
                        .cpu()
                        .detach()
                        .numpy()
                        .reshape(-1, num_tokens * last_dim)
                    )
            else:
                res = model_res[1].cpu().detach().numpy()
            return res

    def apply_fn(features):
        embeddings = []
        for f in features:
            sample_in = f
            if not string_input:
                sample_in = f.decode(FLAGS.text_decodeformat)
            embeddings.append(compute(sample_in))

        return np.stack(embeddings)

    return apply_fn


def load_and_apply_from_textfile():
    if FLAGS.torchhub_type == "image":
        raise app.UsageError(
            "Torch module for images not allowed for text files inputs!"
        )

    if not FLAGS.torchhub_tokenizer_github:
        raise app.UsageError("--torchhub_tokenizer_github has to be specified!")
    if not FLAGS.torchhub_tokenizer_name:
        raise app.UsageError("--torchhub_tokenizer_name has to be specified!")

    return read_from_textfile(get_text_fn(True))


def load_and_apply_from_tfds():
    if not FLAGS.torchhub_tokenizer_github:
        raise app.UsageError("--torchhub_tokenizer_github has to be specified!")
    if not FLAGS.torchhub_tokenizer_name:
        raise app.UsageError("--torchhub_tokenizer_name has to be specified!")

    return read_from_tfds(get_text_fn(False), False, False, False)


def load_and_apply():
    setup()

    if FLAGS.variant == "tfds" and FLAGS.torchhub_type == "text":
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

    if FLAGS.torchhub_args is None:
        model = torch.hub.load(
            FLAGS.torchhub_github, FLAGS.torchhub_name, pretrained=True
        )
    else:
        model = torch.hub.load(
            FLAGS.torchhub_github,
            FLAGS.torchhub_name,
            FLAGS.torchhub_args,
            pretrained=True,
        )
    model.eval()

    new_features = []

    def get_features():
        def hook(model, input, output):
            res = output.cpu().detach().numpy()
            new_features.extend(res.reshape(res.shape[0], -1))

        return hook

    if (
        FLAGS.torchhub_name == "alexnet"
        or FLAGS.torchhub_name == "vgg19"
        or FLAGS.torchhub_name == "vgg16"
    ):
        model.classifier[5].register_forward_hook(get_features())
    elif FLAGS.torchhub_name == "googlenet":
        model.dropout.register_forward_hook(get_features())
    else:
        raise NotImplementedError(
            "Hook for exported representation for model '{}' not implemented yet!".format(
                FLAGS.torchhub_name
            )
        )

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model.to("cuda")

    if FLAGS.torchhub_adjust == "resize":
        adjust = transforms.Resize(FLAGS.torchhub_mininput)
    elif FLAGS.torchhub_adjust == "pad":
        pad_left = (FLAGS.torchhub_mininput - FLAGS.input_width) // 2
        pad_top = (FLAGS.torchhub_mininput - FLAGS.input_height) // 2
        pad_right = FLAGS.torchhub_mininput - FLAGS.input_width - pad_left
        pad_bottom = FLAGS.torchhub_mininput - FLAGS.input_height - pad_top
        adjust = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom))
    elif FLAGS.torchhub_adjust == "mix":
        mid = int(FLAGS.torchhub_mininput * 0.75)
        adjust1 = transforms.Resize(mid)
        pad_left = (FLAGS.torchhub_mininput - mid) // 2
        pad_top = (FLAGS.torchhub_mininput - mid) // 2
        pad_right = FLAGS.torchhub_mininput - mid - pad_left
        pad_bottom = FLAGS.torchhub_mininput - mid - pad_top
        adjust2 = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom))
        adjust = transforms.Compose([adjust1, adjust2])
    else:
        raise NotImplementedError("Adjusting variant not implemented!")

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            adjust,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def fn(images):
        if images.dtype is not np.dtype(np.float32):
            images = images.astype(np.float32)
        images = [preprocess(x) for x in images]
        images = torch.stack(images)
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            images = images.to("cuda")

        with torch.no_grad():
            output = model(images)

    apply_fn_matrices(features, dim, samples, labels, fn, False)

    features = np.array(new_features)
    samples = features.shape[0]
    dim = features.shape[1]

    if len(labels.shape) != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if labels.shape[0] != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features, dim, samples, labels
