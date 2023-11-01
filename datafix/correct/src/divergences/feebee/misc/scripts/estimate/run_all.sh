scripts="scripts/estimate/de_knn.sh scripts/estimate/kde.sh scripts/estimate/lr_model.sh scripts/estimate/onenn_knn.sh scripts/estimate/ghp.sh scripts/estimate/knn.sh scripts/estimate/knn_loo.sh scripts/estimate/knn_extrapolate.sh"

for dataset in cifar10-aggre cifar10-random1 cifar10-random2 cifar10-random3 cifar10-worst cifar100-noisy
do
  for suffix in raw pca_32 pca_64 pca_128 nca_64 alexnet_pt googlenet_pt inception_v3_tf efficientnet_b0_tf efficientnet_b1_tf efficientnet_b2_tf efficientnet_b3_tf efficientnet_b4_tf efficientnet_b5_tf efficientnet_b6_tf efficientnet_b7_tf resetnet_v2_101_tf resetnet_v2_152_tf resetnet_v2_50_tf vgg16_pt vgg19_pt
  do
    for s in $scripts
    do
      bash $s $dataset $suffix 10
    done
  done
done

for dataset in mnist cifar10 cifar100
do
  for suffix in raw pca_32 pca_64 pca_128 nca_64 alexnet_pt googlenet_pt inception_v3_tf efficientnet_b0_tf efficientnet_b1_tf efficientnet_b2_tf efficientnet_b3_tf efficientnet_b4_tf efficientnet_b5_tf efficientnet_b6_tf efficientnet_b7_tf resetnet_v2_101_tf resetnet_v2_152_tf resetnet_v2_50_tf vgg16_pt vgg19_pt
  do
    for s in $scripts
    do
      bash $s $dataset $suffix 10
    done
  done
done

for dataset in imdb sst2
do
  for suffix in bow bow_tfidf bert elmo nnlm50 nnlm50_norm nnlm128 nnlm128_norm use pca_8 pca_16 pca_32 pca_64 pca_128
  do
    for s in $scripts
    do
      bash $s $dataset $suffix 10
    done
  done
done

dataset="yelp"
for suffix in bert elmo nnlm50 nnlm50_norm nnlm128 nnlm128_norm use 
do
  for s in $scripts
  do
    bash $s $dataset $suffix 5
  done
done

dataset="checkerboard_medium"
for suffix in raw
do
  for s in $scripts
  do
    bash $s $dataset $suffix 10
  done
done
