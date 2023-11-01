dataset="checkerboard_medium"
suffix="raw"

mkdir -p "matrices/$dataset/train/"
mkdir -p "matrices/$dataset/test/"

split="train"
outputfolder=matrices/$dataset/$split
mkdir -p $outputfolder

cmd="python generate_checkerboard.py --samples 10000 --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy"
echo "Running $cmd"
$cmd
echo "Done $cmd"

split="test"
outputfolder=matrices/$dataset/$split
mkdir -p $outputfolder

cmd="python generate_checkerboard.py --samples 1000 --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy"
echo "Running $cmd"
$cmd
echo "Done $cmd"
