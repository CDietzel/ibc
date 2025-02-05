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

"""Test a Behavioral Cloning network for pose mapping on an interbotix arm."""
#  pylint: disable=g-long-lambda
import collections
import datetime
import functools
import math
import os
import pickle
from itertools import chain, product, repeat
from pathlib import Path
from typing import OrderedDict

import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags, logging
from tf_agents.policies import policy_saver
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.trajectories import StepType
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import StepType, TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common, example_encoding, example_encoding_dataset

from ibc.data import dataset as x100_dataset_tools
from ibc.environments.block_pushing import (  # pylint: disable=unused-import
    block_pushing,
    block_pushing_discontinuous,
)
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.ibc import tasks
from ibc.ibc.agents import ibc_policy  # pylint: disable=unused-import
from ibc.ibc.eval import eval_env as eval_env_module
from ibc.ibc.train import get_agent as agent_module
from ibc.ibc.train import get_cloning_network as cloning_network_module
from ibc.ibc.train import get_data as data_module
from ibc.ibc.train import get_eval_actor as eval_actor_module
from ibc.ibc.train import get_learner as learner_module
from ibc.ibc.train import get_normalizers as normalizers_module
from ibc.ibc.train import get_sampling_spec as sampling_spec_module
from ibc.ibc.utils import make_video as video_module

flags.DEFINE_string("tag", None, "Tag for the experiment. Appended to the root_dir.")
flags.DEFINE_bool(
    "add_time", False, "If True current time is added to the experiment path."
)
flags.DEFINE_multi_string("gin_file", None, "Paths to the gin-config files.")
flags.DEFINE_multi_string("gin_bindings", None, "Gin binding parameters.")
flags.DEFINE_bool(
    "shared_memory_eval", False, "If true the eval_env uses shared_memory."
)
flags.DEFINE_bool("video", False, "If true, write out one rollout video after eval.")
flags.DEFINE_multi_enum(
    "task",
    None,
    (tasks.IBC_TASKS + tasks.D4RL_TASKS + ["INTERBOTIX"]),
    "If True the reach task is evaluated.",
)
flags.DEFINE_boolean(
    "viz_img", default=False, help="Whether to save out imgs of what happened."
)
flags.DEFINE_bool(
    "skip_eval",
    False,
    "If true the evals are skipped and instead run from " "policy_eval binary.",
)
flags.DEFINE_bool("multi_gpu", False, "If true, run in multi-gpu setting.")

flags.DEFINE_enum("device_type", "gpu", ["gpu", "tpu"], "Where to perform training.")

FLAGS = flags.FLAGS
VIZIER_KEY = "success"


@gin.configurable
def train_eval(
    task=None,
    dataset_path=None,
    root_dir=None,
    # 'ebm' or 'mse' or 'mdn'.
    loss_type=None,
    # Name of network to train. see get_cloning_network.
    network=None,
    # Training params
    batch_size=512,
    num_iterations=20000,
    learning_rate=1e-3,
    decay_steps=100,
    replay_capacity=100000,
    eval_interval=1000,
    eval_loss_interval=100,
    eval_episodes=1,
    fused_train_steps=100,
    sequence_length=2,
    uniform_boundary_buffer=0.05,
    for_rnn=False,
    flatten_action=True,
    dataset_eval_fraction=0.0,
    goal_tolerance=0.02,
    tag=None,
    add_time=False,
    seed=0,
    viz_img=False,
    skip_eval=False,
    num_envs=1,
    shared_memory_eval=False,
    image_obs=False,
    strategy=None,
    # Use this to sweep amount of tfrecords going into training.
    # -1 for 'use all'.
    max_data_shards=-1,
    use_warmup=False,
    folder_num=0,
):
    """Tests a BC agent on the given datasets."""

    tf.random.set_seed(seed)  # SETS SEED TO 0, MAYBE CONFIGURABLE??? DO I CARE?
    if not tf.io.gfile.exists(root_dir):
        tf.io.gfile.makedirs(root_dir)  # MAKE GFILE (PORTRABLE FILESYSTEM ABSTRACTION)
    policy_dir = os.path.join("/home/dietzelcc/Documents/Repos/ibc/ibc/", "models")
    if not tf.io.gfile.exists(policy_dir):
        tf.io.gfile.makedirs(policy_dir)
    output_dir = os.path.join("/home/dietzelcc/Documents/Repos/ibc/ibc/", "output")
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    # Logging.
    # if add_time:
    #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if tag:
        root_dir = os.path.join(root_dir, tag)
    if add_time:
        root_dir = os.path.join(root_dir, str(folder_num))

    # Saving the trained model.
    if tag:
        policy_dir = os.path.join(policy_dir, tag)
    if add_time:
        policy_dir = os.path.join(policy_dir, str(folder_num))

    # Saving the model outputs.
    if tag:
        output_dir = os.path.join(output_dir, tag)
    if add_time:
        output_dir = os.path.join(output_dir, str(folder_num))

    eval_fraction = dataset_eval_fraction
    flatten = flatten_action
    norm_function = None
    seq_len = sequence_length
    buffer_size_per_shard = 100
    deterministic = False
    compress_image = True

    path_to_shards = x100_dataset_tools.get_shards(dataset_path)
    if max_data_shards != -1:
        path_to_shards = path_to_shards[:max_data_shards]
        logging.info("limited to %d shards", max_data_shards)
    if not path_to_shards:
        raise ValueError("No data found at %s" % dataset_path)

    num_eval_shards = int(len(path_to_shards) * eval_fraction)
    num_train_shards = len(path_to_shards) - num_eval_shards
    train_shards = path_to_shards[0:num_train_shards]
    if num_eval_shards > 0:
        eval_shards = path_to_shards[num_train_shards:]

    def filter_episodes(traj):
        """Map TFRecord windows (of adjacent TimeSteps) to single episodes.

        Outputs the last episode within a sample window. It does this by using
        the step_type tensor to break up sequences into single episode sections.
        For example, if step_type is: [FIRST, MID, LAST, FIRST, MID, MID], we
        will return a sample, whos tensor indices are sampled as:
        [3, 3, 3, 3, 4, 5]. So that the 3rd index frame is replicated 3 times to
        across the beginning of the tensor.

        Args:
        traj: Trajectory.

        Returns:
        Trajectory containing filtered sample with only one episode.
        """
        step_types = traj.step_type
        seq_len = tf.cast(tf.shape(step_types)[0], tf.int32)

        # Find the last start frame in the window. e.g. if we have step types
        # [FIRST, MID, LAST, FIRST, MID, MID], we want index 3.
        first_frames = tf.where(step_types == StepType.FIRST)

        if tf.shape(first_frames)[0] == 0:
            # No first frame, return sequence as is.
            inds = tf.range(0, seq_len)
        else:
            ind_start = tf.cast(first_frames[-1, 0], tf.int32)
            if ind_start == 0:
                # Last episode starts on the first frame, return as is.
                inds = tf.range(0, seq_len)
            else:
                # Otherwise, resample so that the last episode's first frame is
                # replicated to the beginning of the sample. In the example above we want:
                # [3, 3, 3, 3, 4, 5].
                inds_start = tf.tile(ind_start[None], ind_start[None])
                inds_end = tf.range(ind_start, seq_len)
                inds = tf.concat([inds_start, inds_end], axis=0)

        def _resample(arr):
            if isinstance(arr, tf.Tensor):
                return tf.gather(arr, inds)
            else:
                return arr  # empty or None

        observation = tf.nest.map_structure(_resample, traj.observation)

        return Trajectory(
            step_type=_resample(traj.step_type),
            action=_resample(traj.action),
            policy_info=_resample(traj.policy_info),
            next_step_type=_resample(traj.next_step_type),
            reward=_resample(traj.reward),
            discount=_resample(traj.discount),
            observation=observation,
        )

    def interleave_func(shard):
        dataset = (
            tf.data.TFRecordDataset(shard, buffer_size=buffer_size_per_shard).cache()
            # .repeat()
        )
        dataset = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)
        return dataset.flat_map(
            lambda window: window.batch(seq_len, drop_remainder=True)
        )

    def _make_dataset(path_to_shards):
        specs = []
        for dataset_file in path_to_shards:
            spec_path = (
                dataset_file + example_encoding_dataset._SPEC_FILE_EXTENSION
            )  # pylint: disable=protected-access
            # Reads tfrecord spec
            dataset_spec = example_encoding_dataset.parse_encoded_spec_from_file(
                spec_path
            )
            specs.append(dataset_spec)
            if not all([dataset_spec == spec for spec in specs]):
                raise ValueError("One or more of the encoding specs do not match.")
        decoder = example_encoding.get_example_decoder(
            specs[0], batched=True, compress_image=compress_image
        )
        dataset = tf.data.Dataset.from_tensor_slices(path_to_shards)
        num_parallel_calls = None if deterministic else len(path_to_shards)
        dataset = dataset.interleave(
            interleave_func,
            deterministic=deterministic,
            cycle_length=len(path_to_shards),
            block_length=1,
            num_parallel_calls=num_parallel_calls,
        )
        dataset = dataset.map(decoder, num_parallel_calls=num_parallel_calls)

        # if for_rnn:
        #     return dataset.map(
        #         filter_episodes_rnn, num_parallel_calls=num_parallel_calls
        #     )
        # else:
        dataset = dataset.map(filter_episodes, num_parallel_calls=num_parallel_calls)

        # Set observation shape.
        def set_shape_obs(traj):
            def set_elem_shape(obs):
                obs_shape = obs.get_shape()
                return tf.ensure_shape(obs, [seq_len] + obs_shape[1:])

            observation = tf.nest.map_structure(set_elem_shape, traj.observation)
            return traj._replace(observation=observation)

        dataset = dataset.map(set_shape_obs, num_parallel_calls=num_parallel_calls)

        # sequence_dataset = (
        #     sequence_dataset.repeat()
        #     .shuffle(replay_capacity)
        #     .batch(batch_size, drop_remainder=True)
        # )
        sequence_dataset = dataset
        return sequence_dataset

    train_data = _make_dataset(train_shards)
    if num_eval_shards > 0:
        eval_data = _make_dataset(eval_shards)
    else:
        eval_data = None

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

    def convert_to_time_step(trajectory):
        step_type = trajectory.step_type
        reward = trajectory.reward
        discount = trajectory.discount
        observation = trajectory.observation

        return TimeStep(
            step_type=tf.expand_dims(tf.cast(step_type[0], tf.int32), 0),
            reward=tf.expand_dims(reward[0], 0),
            discount=tf.expand_dims(discount[0], 0),
            observation={
                "human_pose": tf.expand_dims(observation["human_pose"], 0),
                "action": tf.expand_dims(observation["action"], 0),
            },
        )

    def get_actual_action(trajectory):
        action = trajectory.action
        return action

    time_steps = train_data.map(
        convert_to_time_step, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )

    actual_action_dataset = train_data.map(
        get_actual_action, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )

    # Loading the policy:
    policy = tf.saved_model.load(policy_dir)

    # Doing inference with the policy:
    def get_predicted_action(time_step):
        return policy.action(time_step).action

    # predicted_action_dataset = time_steps.map(
    #     get_predicted_action, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    # )

    predicted_action_list = []
    actual_action_list = []
    actual_observation_list = []

    # for action in predicted_action_dataset:
    #     predicted_action_list.append(action.numpy().tolist()[0])

    for action in actual_action_dataset:
        actual_action_list.append(action[0].numpy().tolist())

    for time_step in time_steps:
        actual_observation_list.append(
            time_step.observation["human_pose"][0][0, 7].numpy().tolist()
        )

        if not predicted_action_list:
            # for action in time_step.observation["action"][0].numpy().tolist():
            action = time_step.observation["action"][0].numpy().tolist()[-1]
            for i in range(seq_len):
                predicted_action_list.append(action)
        prev_action = np.array(predicted_action_list[-seq_len:], dtype=np.float32)
        time_step.observation["action"] = tf.expand_dims(prev_action, 0)
        action = get_predicted_action(time_step)[0].numpy().tolist()
        predicted_action_list.append(action)
    del predicted_action_list[0]  # Doesn't make much sense why it lines up like this
    del predicted_action_list[-(seq_len - 1) :]  # But whatever, it gives best results

    # del predicted_action_list[: (seq_len - 1)]
    # del predicted_action_list[-1]

    # del predicted_action_list[:seq_len]

    output_filename = os.path.splitext(os.path.basename(path_to_shards[0]))[0]
    # dataset_template = os.path.splitext(os.path.basename(dataset_path))[0].split("*")
    # output_filename = output_filename[len(dataset_template[0]) :]
    # output_file = output_file[:-len(dataset_template[-1])]

    joint_names = ["Waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]

    predicted_action_table = pd.DataFrame(predicted_action_list, columns=joint_names)
    actual_action_table = pd.DataFrame(actual_action_list, columns=joint_names)
    actual_observation_table = pd.DataFrame(actual_observation_list)

    _, axes = plt.subplots(nrows=1, ncols=3)
    actual_observation_table.plot(ax=axes[0])
    actual_action_table.plot(ax=axes[1])
    predicted_action_table.plot(ax=axes[2])
    plt.gcf().set_size_inches(10, 5)

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    plt.savefig(output_dir + "/" + output_filename + ".png", dpi=200)
    plt.close("all")

    with open(Path(output_dir) / (output_filename + str(".modulated")), "wb") as f:
        pickle.dump(
            np.array(predicted_action_list), f, protocol=pickle.HIGHEST_PROTOCOL
        )

    # plt.show(block=True)
    pass


def main(_):
    logging.set_verbosity(logging.INFO)

    ebm_params = {
        "gin_file": ["ibc/ibc/configs/interbotix/mlp_ebm_langevin_test.gin"],
        "tag": ["ibc_langevin_test"],
        "network_size": [[512, 4], [256, 8], [128, 16]],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "num_counter_examples": [8, 4],
    }

    # for EBM:
    # hyperparams to train: MLPEBM.width, MLPEBM.depth,
    # ImplicitBCAgent.num_counter_examples,
    # train_eval.learning_rate, train_eval.sequence_length

    mse_params = {
        "gin_file": ["ibc/ibc/configs/interbotix/mlp_mse_test.gin"],
        "tag": ["mse_test"],
        "network_size": [[512, 4], [256, 8], [128, 16]],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "dropout_rate": [0, 0.1],
    }

    seed = 0
    folder_offset = 0

    ebm_keys, ebm_values = zip(*ebm_params.items())
    mse_keys, mse_values = zip(*mse_params.items())
    for i, (keys, values) in enumerate(
        chain(
            zip(repeat(ebm_keys), product(*ebm_values)),
            zip(repeat(mse_keys), product(*mse_values)),
        )
    ):
        for test_file_num in range(1, 6):
            # if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            #     continue
            d = dict(zip(keys, values))
            gin_file = [d["gin_file"]]
            tag = d["tag"]
            bindings = []
            if d["tag"] == "ibc_langevin_test":
                bindings.append(f"MLPEBM.width={d['network_size'][0]}")
                bindings.append(f"MLPEBM.depth={d['network_size'][1]}")
                bindings.append(f"train_eval.learning_rate={d['learning_rate']}")
                bindings.append(
                    f"ImplicitBCAgent.num_counter_examples={d['num_counter_examples']}"
                )
            elif d["tag"] == "mse_test":
                bindings.append(f"MLPMSE.width={d['network_size'][0]}")
                bindings.append(f"MLPMSE.depth={d['network_size'][1]}")
                bindings.append(f"train_eval.learning_rate={d['learning_rate']}")
                bindings.append(f"MLPMSE.rate={d['dropout_rate']}")
            bindings.append(
                f"train_eval.dataset_path='ibc/data/interbotix_test_data/oracle_interbotix_test{test_file_num}_n.tfrecord'"
            )
            bindings.append(f"train_eval.seed={seed}")
            gin.clear_config()
            gin.add_config_file_search_path(os.getcwd())
            gin.parse_config_files_and_bindings(
                gin_file,
                bindings,
                # TODO(coreylynch): This is a temporary
                # hack until we get proper distributed
                # eval working. Remove it once we do.
                skip_unknown=True,
            )

            # For TPU, FLAGS.tpu will be set with a TPU address and FLAGS.use_gpu
            # will be False.
            # For GPU, FLAGS.tpu will be None and FLAGS.use_gpu will be True.
            strategy = strategy_utils.get_strategy(tpu=FLAGS.tpu, use_gpu=FLAGS.use_gpu)

            task = FLAGS.task or gin.REQUIRED
            # If setting this to True, change `my_rangea in mcmc.py to `= range`
            tf.config.experimental_run_functions_eagerly(False)
            train_eval(
                task=task,
                tag=tag,
                add_time=FLAGS.add_time,
                viz_img=FLAGS.viz_img,
                skip_eval=FLAGS.skip_eval,
                shared_memory_eval=FLAGS.shared_memory_eval,
                strategy=strategy,
                folder_num=i + folder_offset,
            )


# def main(_):
#     logging.set_verbosity(logging.INFO)

#     gin.add_config_file_search_path(os.getcwd())
#     gin.parse_config_files_and_bindings(
#         FLAGS.gin_file,
#         FLAGS.gin_bindings,
#         # TODO(coreylynch): This is a temporary
#         # hack until we get proper distributed
#         # eval working. Remove it once we do.
#         skip_unknown=True,
#     )

#     # For TPU, FLAGS.tpu will be set with a TPU address and FLAGS.use_gpu
#     # will be False.
#     # For GPU, FLAGS.tpu will be None and FLAGS.use_gpu will be True.
#     strategy = strategy_utils.get_strategy(tpu=FLAGS.tpu, use_gpu=FLAGS.use_gpu)

#     task = FLAGS.task or gin.REQUIRED
#     # If setting this to True, change `my_rangea in mcmc.py to `= range`
#     tf.config.experimental_run_functions_eagerly(False)

#     train_eval(
#         task=task,
#         tag=FLAGS.tag,
#         add_time=FLAGS.add_time,
#         viz_img=FLAGS.viz_img,
#         skip_eval=FLAGS.skip_eval,
#         shared_memory_eval=FLAGS.shared_memory_eval,
#         strategy=strategy,
#     )


if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))
