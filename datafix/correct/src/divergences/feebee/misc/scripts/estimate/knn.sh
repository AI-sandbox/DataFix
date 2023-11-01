dataset=$1
suffix=$2
runs=$3

split="train"
outputfolder=outputs/$dataset/knn/$split
mkdir -p $outputfolder

for measure in cosine squared_l2
do
  cmd="python estimate.py --method knn --knn_measure $measure --features_train matrices/$dataset/$split/features_$suffix.npy --labels_train matrices/$dataset/$split/labels_$suffix.npy --features_test matrices/$dataset/test/features_$suffix.npy --labels_test matrices/$dataset/test/labels_$suffix.npy --noise_runs $runs -v 1 --output_file $outputfolder/${suffix}_measure_${measure}.csv --knn_subtest 10000 --knn_subtrain 50000"

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
