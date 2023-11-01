tfds=$1
train_split=$2
test_split=$3
feature=$4
module=$5
dataset=$6
suffix=$7

tfds_splits=($train_split $test_split)
output=("train" "test")

for index in 0 1
do
    split=${output[$index]}

    outputfolder=matrices/$dataset/$split
    mkdir -p $outputfolder

    cmd="python export.py --variant tfds --tfds_name $tfds --tfds_split ${tfds_splits[$index]} --tfds_feature $feature --tfds_label label --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --transformations tfhub_module --tfhub_module https://tfhub.dev/google/$module --tfhub_type text -v 1 --batch_size 8"
    echo $cmd
    $cmd
done
