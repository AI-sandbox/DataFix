dataset=$1
suffix_input=$2
nca_dim=$3

suffix="nca_$nca_dim"

split=train
outputfolder=matrices/$dataset/$split
cmd="python export.py --variant matrix --features_matrix $outputfolder/features_$suffix_input.npy --labels_matrix $outputfolder/labels_$suffix_input.npy -v 1 --transformations nca --nca_dim $nca_dim --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy"
echo "Running $cmd"
$cmd
echo "Done $cmd"

nca_fit_features=$outputfolder/features_$suffix_input.npy
nca_fit_labels=$outputfolder/labels_$suffix_input.npy

split=test
outputfolder=matrices/$dataset/$split
cmd="python export.py --variant matrix --features_matrix $outputfolder/features_$suffix_input.npy --labels_matrix $outputfolder/labels_$suffix_input.npy -v 1 --transformations nca --nca_dim $nca_dim --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --nca_fit_features $nca_fit_features --nca_fit_labels $nca_fit_labels"
echo "Running $cmd"
$cmd
echo "Done $cmd"
