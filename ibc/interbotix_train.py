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

"""Train a Behavioral Cloning network for pose mapping on an interbotix arm."""
#  pylint: disable=g-long-lambda
import collections
import datetime
import functools
import math
import os
from itertools import chain, product, repeat
from typing import OrderedDict

import gin
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from tf_agents.policies import policy_saver
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

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
    """Trains a BC agent on the given datasets."""

    # folder_num = 0

    if task is None:
        raise ValueError("task argument must be set.")
    logging.info(("Using task:", task))  # GET TASK NAME

    tf.random.set_seed(seed)  # SETS SEED TO 0, MAYBE CONFIGURABLE??? DO I CARE?
    if not tf.io.gfile.exists(root_dir):
        tf.io.gfile.makedirs(root_dir)  # MAKE GFILE (PORTRABLE FILESYSTEM ABSTRACTION)
    policy_dir = os.path.join("/home/dietzelcc/Documents/Repos/ibc/ibc/", "models")
    if not tf.io.gfile.exists(policy_dir):
        tf.io.gfile.makedirs(policy_dir)

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

    obs_tensor_spec = OrderedDict(
        [
            (
                "human_pose",
                array_spec.BoundedArraySpec(
                    shape=(sequence_length, 36),
                    dtype=np.dtype("float32"),
                    name="observation/human_pose",
                    minimum=-1,
                    maximum=1,
                ),
            ),
            (
                "action",
                array_spec.BoundedArraySpec(
                    shape=(sequence_length, 5),
                    dtype=np.dtype("float32"),
                    name="observation/action",
                    minimum=-2 * math.pi,
                    maximum=2 * math.pi,
                ),
            ),
        ]
    )
    obs_tensor_spec = tensor_spec.from_spec(obs_tensor_spec)
    action_tensor_spec = array_spec.BoundedArraySpec(
        shape=(5,),
        dtype=np.dtype("float32"),
        name="action",
        minimum=-2 * math.pi,
        maximum=2 * math.pi,
    )
    action_tensor_spec = tensor_spec.from_spec(action_tensor_spec)
    time_step_tensor_spec = ts.time_step_spec(obs_tensor_spec)

    # Compute normalization info from training data.
    create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
        dataset_path,
        sequence_length,
        replay_capacity,
        batch_size,
        for_rnn,
        dataset_eval_fraction,
        flatten_action,
    )
    train_data, _ = create_train_and_eval_fns_unnormalized()
    (norm_info, norm_train_data_fn) = normalizers_module.get_normalizers(
        train_data, batch_size, None
    )

    # Create normalized training data.
    if not strategy:
        strategy = tf.distribute.get_strategy()
    per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
    create_train_and_eval_fns = data_module.get_data_fns(
        dataset_path,
        sequence_length,
        replay_capacity,
        per_replica_batch_size,
        for_rnn,
        dataset_eval_fraction,
        flatten_action,
        norm_function=norm_train_data_fn,
        max_data_shards=max_data_shards,
    )
    # Create properly distributed eval data iterator.
    # THIS APPEARS TO DO NOTHING???? JUST RETURNS NONE (COULD BE REMOVED???)
    # MIGHT ONLY BE THIS WAY WITH CERTAIN DATASETS THO
    dist_eval_data_iter = get_distributed_eval_data(create_train_and_eval_fns, strategy)

    # Create normalization layers for obs and action.
    with strategy.scope():
        # Create train step counter.
        # SIMPLE FUNCTION, RETURNS TF VARIABLE
        train_step = train_utils.create_train_step()

        # Define action sampling spec.
        # DEFINES THE MINIMUM AND MAXIMUM VALUES FOR THE ACTIONS IN THE DATASET
        # WHAT IS THE DIFFERENCE BETWEEN THIS AND action_tensor_spec???
        action_sampling_spec = sampling_spec_module.get_sampling_spec(
            action_tensor_spec,
            min_actions=norm_info.min_actions,
            max_actions=norm_info.max_actions,
            uniform_boundary_buffer=uniform_boundary_buffer,
            act_norm_layer=norm_info.act_norm_layer,
        )

        # This is a common opportunity for a bug, having the wrong sampling min/max
        # so log this.
        logging.info(("Using action_sampling_spec:", action_sampling_spec))

        # Define keras cloning network.
        # THIS IS WHERE THE MAGIC HAPPENS!!!!!!!!!!!!
        # THE ACTUAL BEHAVIORAL CLONING NETWORK THAT DOES THE THING
        # DEFINITELY STEAL THIS CODE
        cloning_network = cloning_network_module.get_cloning_network(
            network,
            obs_tensor_spec,
            action_tensor_spec,
            norm_info.obs_norm_layer,
            norm_info.act_norm_layer,
            sequence_length,
            norm_info.act_denorm_layer,
        )

        # Define tfagent.
        # MAKES THE BEHAVIORAL CLONING AGENT, PROVIDES LOSS FUNCTION AND
        # COUNTEREXAMPLE GENERATOR. ALSO STEAL THIS CODE
        agent = agent_module.get_agent(
            loss_type,
            time_step_tensor_spec,
            action_tensor_spec,
            action_sampling_spec,
            norm_info.obs_norm_layer,
            norm_info.act_norm_layer,
            norm_info.act_denorm_layer,
            learning_rate,
            use_warmup,
            cloning_network,
            train_step,
            decay_steps,
        )

        # Define bc learner.
        # HANDLES CHECKPOINTING THE NETWORK OVER THE TRAINING PROCESS
        bc_learner = learner_module.get_learner(
            loss_type,
            root_dir,
            agent,
            train_step,
            create_train_and_eval_fns,
            fused_train_steps,
            strategy,
        )

        get_eval_loss = tf.function(agent.get_eval_loss)

        # Get summary writer for aggregated metrics.
        aggregated_summary_dir = os.path.join(root_dir, "eval")
        summary_writer = tf.summary.create_file_writer(
            aggregated_summary_dir, flush_millis=10000
        )
    logging.info("Saving operative-gin-config.")
    with tf.io.gfile.GFile(
        os.path.join(root_dir, "operative-gin-config.txt"), "wb"
    ) as f:
        f.write(gin.config_str())

    # Main train and eval loop.
    while train_step.numpy() < num_iterations:
        # Run bc_learner for fused_train_steps.
        # THIS IS THE CORE FUNCTION THAT RUNS THE TRAINING STEPS
        # DEFINITELY USE THIS
        training_step(agent, bc_learner, fused_train_steps, train_step)

        if (
            dist_eval_data_iter is not None
            and train_step.numpy() % eval_loss_interval == 0
        ):
            # Run a validation step.
            validation_step(dist_eval_data_iter, bc_learner, train_step, get_eval_loss)

    # Saving the policy:
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)

    # Finish writing train/eval summary:
    summary_writer.flush()


def training_step(agent, bc_learner, fused_train_steps, train_step):
    """Runs bc_learner for fused training steps."""
    reduced_loss_info = None
    if not hasattr(agent, "ebm_loss_type") or agent.ebm_loss_type != "cd_kl":
        reduced_loss_info = bc_learner.run(iterations=fused_train_steps)
    else:
        for _ in range(fused_train_steps):
            # I think impossible to do this inside tf.function.
            agent.cloning_network_copy.set_weights(agent.cloning_network.get_weights())
            reduced_loss_info = bc_learner.run(iterations=1)

    if reduced_loss_info:
        # Graph the loss to compare losses at the same scale regardless of
        # number of devices used.
        with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(True):
            tf.summary.scalar("reduced_loss", reduced_loss_info.loss, step=train_step)


def validation_step(dist_eval_data_iter, bc_learner, train_step, get_eval_loss_fn):
    """Runs a validation step."""
    losses_dict = get_eval_loss_fn(next(dist_eval_data_iter))

    with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(True):
        common.summarize_scalar_dict(
            losses_dict, step=train_step, name_scope="Eval_Losses/"
        )


def evaluation_step(eval_episodes, eval_env, eval_actor, name_scope_suffix=""):
    """Evaluates the agent in the environment."""
    logging.info("Evaluating policy.")
    with tf.name_scope("eval" + name_scope_suffix):
        # This will eval on seeds:
        # [0, 1, ..., eval_episodes-1]
        for eval_seed in range(eval_episodes):
            eval_env.seed(eval_seed)
            eval_actor.reset()  # With the new seed, the env actually needs reset.
            eval_actor.run()

        eval_actor.log_metrics()
        eval_actor.write_metric_summaries()
    return eval_actor.metrics


def get_distributed_eval_data(data_fn, strategy):
    """Gets a properly distributed evaluation data iterator."""
    _, eval_data = data_fn()
    dist_eval_data_iter = None
    if eval_data:
        dist_eval_data_iter = iter(
            strategy.distribute_datasets_from_function(lambda: eval_data)
        )
    return dist_eval_data_iter


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
            "train_eval.dataset_path='ibc/data/interbotix_data/oracle_interbotix*.tfrecord'"
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
