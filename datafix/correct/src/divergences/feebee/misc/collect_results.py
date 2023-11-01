from absl import app
from absl import flags
from absl import logging
import os
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "outputs", "Path to the matrices directory")
flags.DEFINE_string(
    "output_file", "outputs/results.csv", "Output file. None to not store results"
)
flags.DEFINE_list("datasets", "", "List of datasets")

CSV_SUFFIX = ".csv"


def main(argv):
    base_path = FLAGS.path

    df = None

    for dataset in sorted(os.listdir(base_path)):
        if not os.path.isdir(os.path.join(base_path, dataset)):
            continue
        if dataset not in FLAGS.datasets:
            continue
        print(dataset)

        for method in sorted(os.listdir(os.path.join(base_path, dataset))):
            for split in sorted(os.listdir(os.path.join(base_path, dataset, method))):
                print("  ", split)

                path = os.path.join(base_path, dataset, method, split)

                suffix = {}

                files = sorted(os.listdir(path))

                for f in files:
                    if f.endswith(CSV_SUFFIX):
                        identifier = f[: -len(CSV_SUFFIX)]
                        df_read = pd.read_csv(os.path.join(path, f))
                        df_read["dataset"] = dataset
                        df_read["identifier"] = identifier
                        if df is None:
                            df = df_read
                        else:
                            df = df.append(df_read)

    if FLAGS.output_file and df is not None:
        df.to_csv(FLAGS.output_file)


if __name__ == "__main__":
    app.run(main)
