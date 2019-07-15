#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/mnt/hgfs/ubuntushared/retrieve_lsh/data/facescrub
DATA=.
TOOLS=/home/ling/workspace/3rd/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/facescrub_train_leveldb \
  $DATA/mean2.binaryproto

echo "Done."
