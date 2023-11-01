module=$1
suffix=$2
filepath=$3

dataset=yelp
feature=text
label=stars

split=train
outputfolder=matrices/$dataset/$split
mkdir -p $outputfolder
start=0
end=500000
cmd="python export.py --variant textfile --file_json $filepath --file_dict_features $feature --file_dict_labels $label --file_startindex $start --file_endindex $end --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --transformations tfhub_module --tfhub_module https://tfhub.dev/google/$module --tfhub_type text -v 1 --batch_size 16"
echo $cmd
$cmd

split=test
outputfolder=matrices/$dataset/$split
mkdir -p $outputfolder
start=500000
end=550000
cmd="python export.py --variant textfile --file_json $filepath --file_dict_features $feature --file_dict_labels $label --file_startindex $start --file_endindex $end --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --transformations tfhub_module --tfhub_module https://tfhub.dev/google/$module --tfhub_type text -v 1 --batch_size 16"
echo $cmd
$cmd
