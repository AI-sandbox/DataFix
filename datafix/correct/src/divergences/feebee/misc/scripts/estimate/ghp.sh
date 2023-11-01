dataset=$1
suffix=$2
runs=$3

split="test"
outputfolder=outputs/$dataset/ghp/$split
mkdir -p $outputfolder

cmd="python estimate.py --method ghp --features_train matrices/$dataset/$split/features_$suffix.npy --labels_train matrices/$dataset/$split/labels_$suffix.npy --noise_runs $runs -v 1 --output_file $outputfolder/$suffix.csv" 

if [ -f "$outputfolder/_lsf.$suffix" ]
then
  echo "File $outputfolder/_lsf.$suffix esits, skipping."
else
  echo $cmd
  bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o $outputfolder/_lsf.$suffix $cmd
fi
