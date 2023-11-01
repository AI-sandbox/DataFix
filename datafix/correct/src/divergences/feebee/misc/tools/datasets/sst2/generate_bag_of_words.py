import tensorflow_datasets as tfds
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os

os.makedirs("matrices/sst2/train/", exist_ok=True)
os.makedirs("matrices/sst2/test/", exist_ok=True)


corpus = []
labels = []

for split in ["train", "validation"]:
    # load dataset from TF
    print("processing " + split + " split...")
    data_samples = tfds.load("glue/sst2", split=split)  # v0.0.2

    for entry in tfds.as_numpy(data_samples):
        # preprocess samples
        raw_text = entry["sentence"].decode("utf-8")
        # samples are already cleaned and tokenized
        corpus.append(raw_text)

        # append label
        labels.append(entry["label"])

print("samples:", len(corpus))

# use simple word counts based on dataset vocabulary
vectorizer = CountVectorizer()
# default configuration tokenizes the string by extracting words of at least 2 letters
bow_matrix = vectorizer.fit_transform(corpus)
baseline_features = bow_matrix.toarray()

# get vocabulary details
# print(vectorizer.vocabulary_.items())
print("vocab size:", baseline_features.shape)

# separate train and test split again
train_baseline = baseline_features[:67349]
test_baseline = baseline_features[67349:]

train_labels = labels[:67349]
test_labels = labels[67349:]

# save samples to files
np.save("matrices/sst2/train/features_bow.npy", train_baseline)
np.save("matrices/sst2/test/features_bow.npy", test_baseline)

np.save("matrices/sst2/train/labels_bow.npy", train_labels)
np.save("matrices/sst2/test/labels_bow.npy", test_labels)
