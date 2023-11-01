tfds=imdb_reviews
train_split=train
test_split=test
feature=text
dataset=imdb
suffix=bow_tfidf

mkdir -p "matrices/$dataset/train/"
mkdir -p "matrices/$dataset/test/"

sfolder="scripts/export"

# BoW
cmd="python tools/datasets/imdb/generate_bag_of_words.py"
$cmd

# BoW-TFIDF
cmd="python tools/datasets/imdb/generate_bag_of_words_tfidf.py"
$cmd

# PCA 8 / 16 / 32 / 64
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 8"
$cmd
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 16"
$cmd
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 32"
$cmd
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 64"
$cmd

# TF Hub
suffix=elmo
module=elmo/3
cmd="bash $sfolder/export_text_tfds_tfhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd

suffix=nnlm50
module=nnlm-en-dim50/2
cmd="bash $sfolder/export_text_tfds_tfhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd

suffix=nnlm50_norm
module=nnlm-en-dim50-with-normalization/2
cmd="bash $sfolder/export_text_tfds_tfhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd

suffix=nnlm128
module=nnlm-en-dim128/2
cmd="bash $sfolder/export_text_tfds_tfhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd

suffix=nnlm128_norm
module=nnlm-en-dim128-with-normalization/2
cmd="bash $sfolder/export_text_tfds_tfhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd

suffix=use
module=universal-sentence-encoder/4
cmd="bash $sfolder/export_text_tfds_tfhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd

# PyTorch Hub
suffix=bert
module=bert-base-uncased
cmd="bash $sfolder/export_text_tfds_torchhub.sh $tfds $train_split $test_split $feature $module $dataset $suffix"
$cmd
