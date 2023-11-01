tfds=$1
train_split=$2
test_split=$3
feature=$4
dataset=$5
suffix=$6
size=$7

tfds_splits=($train_split $test_split)
output=("train" "test")

for index in 0 1
do
    split=${output[$index]}

    outputfolder=matrices/$dataset/$split
    mkdir -p $outputfolder

    cmd="python export.py --variant tfds --tfds_name $tfds --tfds_split ${tfds_splits[$index]} --tfds_feature $feature --tfds_label label --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy -v 1 --resize_height $size --resize_width $size"
    echo "Running $cmd"
    $cmd
    echo "Done $cmd"
done
