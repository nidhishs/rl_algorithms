# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sac.buffer import ReplayBuffer
from sac.networks import MLPActor, MLPCritic
from sac.agent import SACAgent
from torch.utils.tensorboard import SummaryWriter
from config import sac_parse_args
from logger import Logger


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        
        return env

    return thunk

def to_tensor(x, device="cuda"):
    return torch.tensor(x, dtype=torch.float32, device=device)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


if __name__ == "__main__":
    args = sac_parse_args()
    run_name = f"{args.env_id}__{int(time.time())}"
    Logger("rl-benchmarks", f"./log/sac/{run_name}")
    
    Logger.get_logger().tb.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)]
    )
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = SACAgent(
        envs.single_observation_space.shape, envs.single_action_space.shape, [256, 256], [256, 256], args.gamma, args.tau, args.q_lr,
        args.policy_lr, LOG_STD_MIN, LOG_STD_MAX, args.policy_frequency, args.target_network_frequency
    ).to(device)

    rb = ReplayBuffer(
        args.buffer_size, envs.single_observation_space.shape, envs.single_action_space.shape, num_envs=args.num_envs, device=device
    )
    start_time = time.time()

    
    obs, info = envs.reset()
    for global_step in range(1, args.total_timesteps+1):

        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            actions = agent.act(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        next_obs, rewards, termination, truncation, info = envs.step(actions)
        rb.add(to_tensor(obs), to_tensor(actions), to_tensor(rewards), to_tensor(next_obs), to_tensor(termination))

        obs = next_obs

        if global_step % args.log_frequency == 0 or global_step == args.total_timesteps:
            Logger.get_logger().info(
                f"{'Env Interaction':<20}" + " : " + f"[ {f'{global_step:6d}/{args.total_timesteps:6d}':<15} ]"
            )
        
        if "episode" in info:
            episodic_rewards = [x if x_ else np.nan for x, x_ in zip(info["episode"]["r"], info["_episode"])]
            Logger.get_logger().info(
                f"{'Episodic Reward':<20}" + " : " + f"[ {f'{global_step:6d}/{args.total_timesteps:6d}':<15} ]" +
                " : " + f"({', '.join([f'{r:.2f}' for r in episodic_rewards])})"
            )
            Logger.get_logger().tb.add_scalars(
                    "SACAgent/Episodic Reward", {f"Env {i}": episodic_rewards[i] for i in range(args.num_envs)},
                    global_step
                )
                
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            agent.optimize(data, global_step)

    envs.close()