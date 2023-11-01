tfds=$1
train_split=$2
test_split=$3
feature=$4
model=$5
dataset=$6
suffix=$7

tfds_splits=($train_split $test_split)
output=("train" "test")

for index in 0 1
do
    split=${output[$index]}

    outputfolder=matrices/$dataset/$split
    mkdir -p $outputfolder

    cmd="python export.py --variant tfds --tfds_name $tfds --tfds_split ${tfds_splits[$index]} --tfds_feature $feature --tfds_label label --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --transformations torchhub_model --torchhub_github huggingface/pytorch-transformers --torchhub_name model --torchhub_args $model --torchhub_tokenizer_github huggingface/pytorch-transformers --torchhub_tokenizer_name tokenizer --torchhub_tokenizer_args $model --torchhub_type text --torchhub_maxinputlength 500 -v 1 --tfds_datadir /mnt/ds3lab-scratch/rengglic/tensorflow_datasets"
    echo $cmd
    $cmd
done
