tfds=cifar100
train_split=train
test_split=test
feature=image
dataset=cifar100
suffix=raw
size=32
channels=3

mkdir -p "matrices/$dataset/train/"
mkdir -p "matrices/$dataset/test/"

sfolder="scripts/export"

# RAW
cmd="bash $sfolder/export_images_tfds_raw.sh $tfds $train_split $test_split $feature $dataset $suffix $size"
$cmd

# PCA 32 / 64 / 128
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 32"
$cmd
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 64"
$cmd
cmd="bash $sfolder/export_all_pca.sh $dataset $suffix 128"
$cmd

# NCA 64
cmd="bash $sfolder/export_all_nca.sh $dataset $suffix 64"
$cmd

# TF Hub
suffix=resetnet_v2_50_tf
module=imagenet/resnet_v2_50/feature_vector/3
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=resetnet_v2_101_tf
module=imagenet/resnet_v2_101/feature_vector/3
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=resetnet_v2_152_tf
module=imagenet/resnet_v2_152/feature_vector/3
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=inception_v3_tf
module=imagenet/inception_v3/feature_vector/3
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b0_tf
module=efficientnet/b0/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b1_tf
module=efficientnet/b1/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b2_tf
module=efficientnet/b2/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b3_tf
module=efficientnet/b3/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b4_tf
module=efficientnet/b4/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b5_tf
module=efficientnet/b5/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b6_tf
module=efficientnet/b6/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=efficientnet_b7_tf
module=efficientnet/b7/feature-vector/1
cmd="bash $sfolder/export_images_raw_tfhub.sh $dataset $suffix $module $size $channels"
$cmd

# PyTorch Hub
suffix=alexnet_pt
module=alexnet
cmd="bash $sfolder/export_images_raw_pytorchhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=googlenet_pt
module=googlenet
cmd="bash $sfolder/export_images_raw_pytorchhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=vgg16_pt
module=vgg16
cmd="bash $sfolder/export_images_raw_pytorchhub.sh $dataset $suffix $module $size $channels"
$cmd

suffix=vgg19_pt
module=vgg19
cmd="bash $sfolder/export_images_raw_pytorchhub.sh $dataset $suffix $module $size $channels"
$cmd
