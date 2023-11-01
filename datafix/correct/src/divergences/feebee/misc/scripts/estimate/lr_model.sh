dataset=$1
suffix=$2
runs=$3

split="train"
for l2 in 0.0 0.01 0.1
do
  for lr in 0.001 0.01 0.1
  do
    outputfolder=outputs/$dataset/lr_model/$split
    mkdir -p $outputfolder

    cmd="python estimate.py --method lr_model --features_train matrices/$dataset/train/features_$suffix.npy --labels_train matrices/$dataset/train/labels_$suffix.npy --features_test matrices/$dataset/test/features_$suffix.npy --labels_test matrices/$dataset/test/labels_$suffix.npy --noise_runs $runs -v 1 --l2_regs $l2 --sgd_lrs $lr --output_file $outputfolder/${suffix}_l2_${l2}_lr_${lr}.csv" 

    if [ -f "$outputfolder/_lsf.${suffix}_l2_${l2}_lr_${lr}" ]
    then
      echo "File $outputfolder/_lsf.${suffix}_l2_${l2}_lr_${lr} esits, skipping."
    else
      echo $cmd
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o $outputfolder/_lsf.${suffix}_l2_${l2}_lr_${lr} $cmd
    fi
  done
done
