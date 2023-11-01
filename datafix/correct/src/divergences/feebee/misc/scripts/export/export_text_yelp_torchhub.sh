model=$1
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
cmd="python export.py --variant textfile --file_json $filepath --file_dict_features $feature --file_dict_labels $label --file_startindex $start --file_endindex $end --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --transformations torchhub_model --torchhub_github huggingface/pytorch-transformers --torchhub_name model --torchhub_args $model --torchhub_tokenizer_github huggingface/pytorch-transformers --torchhub_tokenizer_name tokenizer --torchhub_tokenizer_args $model --torchhub_type text --torchhub_maxinputlength 500 -v 1"
echo $cmd
$cmd

split=test
outputfolder=matrices/$dataset/$split
mkdir -p $outputfolder
start=500000
end=550000
cmd="python export.py --variant textfile --file_json $filepath --file_dict_features $feature --file_dict_labels $label --file_startindex $start --file_endindex $end --export_features $outputfolder/features_$suffix.npy --export_labels $outputfolder/labels_$suffix.npy --transformations torchhub_model --torchhub_github huggingface/pytorch-transformers --torchhub_name model --torchhub_args $model --torchhub_tokenizer_github huggingface/pytorch-transformers --torchhub_tokenizer_name tokenizer --torchhub_tokenizer_args $model --torchhub_type text --torchhub_maxinputlength 500 -v 1"
echo $cmd
$cmd
