dataset=$1
suffix=$2
module=$3
size=$4
channels=$5

for split in "test" "train"
do
    outputfolder=matrices/$dataset/$split
    mkdir -p $outputfolder

    cmd="python export.py --variant matrix --features_matrix $outputfolder/features_raw.npy --labels_matrix $outputfolder/labels_raw.npy -v 1 --input_height $size --input_width $size --input_channels $channels --transformations torchhub_model --torchhub_github pytorch/vision:v0.5.0 --torchhub_name $module --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy"
    echo "Running $cmd"
    $cmd
    echo "Done $cmd"
done
