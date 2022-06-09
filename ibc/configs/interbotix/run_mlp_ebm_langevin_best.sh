#!/bin/bash

python3 ibc/ibc/interbotix_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/interbotix/mlp_ebm_langevin.gin \
  --task=INTERBOTIX \
  --tag=ibc_langevin \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/interbotix_data/oracle_push*.tfrecord'" \
  --video
