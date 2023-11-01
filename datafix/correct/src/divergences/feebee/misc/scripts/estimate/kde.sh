dataset=$1
suffix=$2
runs=$3

split="test"
for beta in 0.0025 0.05 0.1 0.25 0.5
do
  outputfolder=outputs/$dataset/kde/$split
  mkdir -p $outputfolder

  cmd="python estimate.py --method kde --features_train matrices/$dataset/$split/features_$suffix.npy --labels_train matrices/$dataset/$split/labels_$suffix.npy --noise_runs $runs -v 1 --kde_bandwidth $beta --output_file $outputfolder/${suffix}_beta_$beta.csv" 

  if [ -f "$outputfolder/_lsf.${suffix}_beta_$beta" ]
  then
    echo "File $outputfolder/_lsf.${suffix}_beta_$beta esits, skipping."
  else
    echo $cmd
    bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o $outputfolder/_lsf.${suffix}_beta_$beta $cmd
  fi
done
