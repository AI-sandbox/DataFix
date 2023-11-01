dataset=$1
suffix_input=$2
pca_dim=$3

suffix="pca_$pca_dim"

split=train
outputfolder=matrices/$dataset/$split
cmd="python export.py --variant matrix --features_matrix $outputfolder/features_$suffix_input.npy --labels_matrix $outputfolder/labels_$suffix_input.npy -v 1 --transformations pca --pca_dim $pca_dim --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy"
echo "Running $cmd"
$cmd
echo "Done $cmd"

pca_matrix_path=$outputfolder/features_$suffix_input.npy

split=test
outputfolder=matrices/$dataset/$split
cmd="python export.py --variant matrix --features_matrix $outputfolder/features_$suffix_input.npy --labels_matrix $outputfolder/labels_$suffix_input.npy -v 1 --transformations pca --pca_dim $pca_dim --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --pca_fit_matrix_path $pca_matrix_path"
echo "Running $cmd"
$cmd
echo "Done $cmd"
