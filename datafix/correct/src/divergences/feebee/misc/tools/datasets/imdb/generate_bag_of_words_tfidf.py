import tensorflow_datasets as tfds
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.makedirs("matrices/imdb/train/", exist_ok=True)
os.makedirs("matrices/imdb/test/", exist_ok=True)


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    clean_text = re.sub(cleanr, "", raw_html)
    return clean_text


corpus = []
labels = []

for split in ["train", "test"]:
    # load dataset from TF
    print("processing " + split + " split...")
    data_samples = tfds.load("imdb_reviews", split=split)

    for entry in tfds.as_numpy(data_samples):
        # preprocess samples
        raw_text_html = entry["text"].decode("utf-8")
        raw_text = clean_html(raw_text_html)
        corpus.append(raw_text)

        # append label
        labels.append(entry["label"])

print("samples:", len(corpus))

# use TF-IDF weighting on word counts
vectorizer = TfidfVectorizer()
# default configuration tokenizes the string by extracting words of at least 2 letters
bow_matrix = vectorizer.fit_transform(corpus)
baseline_features = bow_matrix.toarray()

# get vocabulary details
# print(vectorizer.vocabulary_.items())
print("vocab size:", baseline_features.shape)

# separate train and test split again
train_baseline = baseline_features[:25000]
test_baseline = baseline_features[25000:]

train_labels = labels[:25000]
test_labels = labels[25000:]

# save samples to files
np.save("matrices/imdb/train/features_bow_tfidf.npy", train_baseline)
np.save("matrices/imdb/test/features_bow_tfidf.npy", test_baseline)

np.save("matrices/imdb/train/labels_bow_tfidf.npy", train_labels)
np.save("matrices/imdb/test/labels_bow_tfidf.npy", test_labels)
