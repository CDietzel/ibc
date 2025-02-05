# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines training/eval data iterators."""
#  pylint: disable=g-long-lambda
from ibc.data import dataset as x100_dataset_tools
import tensorflow as tf


def get_data_fns(
    dataset_path, # 'ibc/data/interbotix_data/oracle_interbotix*.tfrecord'
    sequence_length, # 2
    replay_capacity, # 10000
    batch_size, # 512
    for_rnn, # False
    dataset_eval_fraction, # 0.0
    flatten, # True
    norm_function=None, # None
    max_data_shards=-1, # -1
):
    """Gets train and eval datasets."""

    # Helper function for creating train and eval data.
    def create_train_and_eval_fns():
        train_data, eval_data = x100_dataset_tools.create_sequence_datasets(
            dataset_path,
            sequence_length,
            replay_capacity,
            batch_size,
            for_rnn=for_rnn,
            eval_fraction=dataset_eval_fraction,
            max_data_shards=max_data_shards,
        )

        def flatten_and_cast_action(action):
            flat_actions = tf.nest.flatten(action)
            flat_actions = [tf.cast(a, tf.float32) for a in flat_actions]
            return tf.concat(flat_actions, axis=-1)

        # def flatten_action(action):
        #     flat_actions = tf.nest.flatten(action)
        #     # flat_actions = [tf.cast(a, tf.float32) for a in flat_actions]
        #     return tf.concat(flat_actions, axis=-1)

        if flatten:
            train_data = train_data.map(
                lambda trajectory: trajectory._replace(
                    action=flatten_and_cast_action(trajectory.action)
                )
            )

            if eval_data:
                eval_data = eval_data.map(
                    lambda trajectory: trajectory._replace(
                        action=flatten_and_cast_action(trajectory.action)  # Unsure if this
                        # should be a call to flatten_action or flatten_and_cast_action
                    )
                )

        # We predict 'many-to-one' observations -> action.
        train_data = train_data.map(
            lambda trajectory: (
                (trajectory.observation, trajectory.action[:, -1, Ellipsis]),
                (),
            )
        )
        if eval_data:
            eval_data = eval_data.map(
                lambda trajectory: (
                    (trajectory.observation, trajectory.action[:, -1, Ellipsis]),
                    (),
                )
            )

        if norm_function:
            train_data = train_data.map(norm_function)
            if eval_data:
                eval_data = eval_data.map(norm_function)

        return train_data, eval_data

    return create_train_and_eval_fns
