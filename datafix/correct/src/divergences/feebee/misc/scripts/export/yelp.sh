dataset=yelp
filepath="~/Downloads/yelp_academic_dataset_review.json"

mkdir -p "matrices/$dataset/train/"
mkdir -p "matrices/$dataset/test/"

sfolder="scripts/export"

# TF Hub
suffix=elmo
module=elmo/3
cmd="bash $sfolder/export_text_yelp_tfhub.sh $module $suffix $filepath"
$cmd

suffix=nnlm50
module=nnlm-en-dim50/2
cmd="bash $sfolder/export_text_yelp_tfhub.sh $module $suffix $filepath"
$cmd

suffix=nnlm50_norm
module=nnlm-en-dim50-with-normalization/2
cmd="bash $sfolder/export_text_yelp_tfhub.sh $module $suffix $filepath"
$cmd

suffix=nnlm128
module=nnlm-en-dim128/2
cmd="bash $sfolder/export_text_yelp_tfhub.sh $module $suffix $filepath"
$cmd

suffix=nnlm128_norm
module=nnlm-en-dim128-with-normalization/2
cmd="bash $sfolder/export_text_yelp_tfhub.sh $module $suffix $filepath"
$cmd

suffix=use
module=universal-sentence-encoder/4
cmd="bash $sfolder/export_text_yelp_tfhub.sh $module $suffix $filepath"
$cmd

# PyTorch Hub
suffix=bert
module=bert-base-uncased
cmd="bash $sfolder/export_text_yelp_torchhub.sh $module $suffix $filepath"
$cmd
