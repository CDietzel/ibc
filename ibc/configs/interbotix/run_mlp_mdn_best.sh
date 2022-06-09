#!/bin/bash

python3 ibc/ibc/interbotix_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/interbotix/mlp_mdn_best.gin \
  --task=INTERBOTIX \
  --tag=mdn \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/interbotix_data/oracle_push*.tfrecord'" \
  --video
