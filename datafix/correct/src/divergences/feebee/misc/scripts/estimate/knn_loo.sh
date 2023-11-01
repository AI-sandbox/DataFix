dataset=$1
suffix=$2
runs=$3

split="test"
outputfolder=outputs/$dataset/knn_loo/$split
mkdir -p $outputfolder

for measure in cosine squared_l2
do
  cmd="python estimate.py --method knn_loo --knn_measure $measure --features_train matrices/$dataset/$split/features_$suffix.npy --labels_train matrices/$dataset/$split/labels_$suffix.npy --noise_runs $runs -v 1 --output_file $outputfolder/${suffix}_measure_${measure}.csv"

  if [ -f "$outputfolder/_lsf.${suffix}_measure_${measure}" ]
  then
    echo "File $outputfolder/_lsf.${suffix}_measure_${measure} esits, skipping."
  else
    i=1
    until [ $i -gt 10 ]
    do
      cmd="$cmd --knn_k $i"
      ((i=i+1))
    done

    echo $cmd
    bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o $outputfolder/_lsf.${suffix}_measure_${measure} $cmd
  fi
done
