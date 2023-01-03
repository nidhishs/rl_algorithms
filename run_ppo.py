import time

import gymnasium as gym
import numpy as np
import torch

import argparse
from logger import Logger
from ppo.agent import PPOAgent
from ppo.core import Buffer, ICM
from ppo.networks import MLPGaussianActorCritic, MLPCategoricalActorCritic

def main():
    args = ppo_parse_args()
    run_name = f"{args.env_id}_{int(time.time())}"
    Logger("rl-benchmarks", f"./log/ppo/{run_name}")

    Logger.get().tb.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env setup
    envs = get_env(args.env_id, args.num_envs)

    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        ac = MLPCategoricalActorCritic(
            np.prod(envs.single_observation_space.shape), envs.single_action_space.n, args.actor_units, args.critic_units,
        )
    elif isinstance(envs.single_action_space, gym.spaces.Box):
        ac = MLPGaussianActorCritic(
            np.prod(envs.single_observation_space.shape), np.prod(envs.single_action_space.shape), args.actor_units, args.critic_units,
        )
    else:
        raise NotImplementedError

    icm = ICM(
        ac.state_dim, ac.action_dim, args.icm_units,
        alpha=args.icm_reward_factor, beta=args.icm_loss_factor
    )
    agent = PPOAgent(
        actor_critic=ac, icm=icm if args.icm else None, policy_lr=args.policy_lr, icm_lr=args.icm_lr, lr_decay_steps=args.num_updates,
        clip_coeff=args.clip_coeff, value_coeff=args.value_coeff, entropy_coeff=args.ent_coeff,
        max_grad_norm=args.max_grad_norm, seed=args.seed
    ).to(args.device)
    buffer = Buffer(
        max_size=args.num_steps, num_envs=args.num_envs, state_dim=envs.single_observation_space.shape,
        action_dim=envs.single_action_space.shape, device=args.device
    )

    global_step = 0
    curr_state, _ = envs.reset(seed=args.seed)
    curr_termination = np.zeros(args.num_envs, dtype=bool)

    for update in range(1, args.num_updates + 1):
        for step in range(1, args.num_steps + 1):
            global_step += args.num_envs

            action, log_prob, _, value = agent.act(to_tensor(curr_state, args.device))
            next_state, curr_reward, next_termination, next_truncation, info = envs.step(action.cpu().numpy())
            buffer.store(
                to_tensor(curr_state, args.device), action, to_tensor(curr_reward, args.device),
                to_tensor(curr_termination, args.device), to_tensor(next_state, args.device), value.flatten(), log_prob
            )

            curr_state = next_state
            curr_termination = next_termination

            if step % args.log_frequency == 0 or step == args.num_steps:
                Logger.get().info(
                    f"{'Trajectory Rollout':<20}" + " : " + f"[ {f'{update:3d}/{args.num_updates} • {step:04d}/{args.num_steps:04d}':<15} ]"
                )

            if "final_info" in info:
                filtered = [x for (x, x_) in zip(info["final_info"], info["_final_info"]) if (x_ and "episode" in x)]
                if filtered:
                    ep_rewards = [x["episode"]["r"].item() for x in filtered]
                    ep_lengths = [x["episode"]["l"].item() for x in filtered]
                    Logger.get().info(
                        f"{'Avg Episodic Rew':<20}" + " : " + f"[ {f'{update:3d}/{args.num_updates} • {step:04d}/{args.num_steps:04d}':<15} ]" +
                        " : " + f"{np.mean(ep_rewards):.3f}"
                    )
                    Logger.get().tb.add_scalar("PPOAgent/reward", np.mean(ep_rewards), global_step)

        last_value = agent.critic(to_tensor(curr_state, args.device))
        batch = agent.compute_advantages_and_returns(
            buffer.get(), last_value.flatten(), to_tensor(curr_termination, args.device), args.gamma, args.gae_lambda
        )
        agent.learn(global_step, batch, args.num_minibatches, args.policy_epochs, args.icm_epochs, args.target_kl)
        buffer.reset()

    envs.close()
    Logger.get().tb.close()

def get_env(env_id, num_envs, gamma=0.99):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    envs = gym.vector.SyncVectorEnv([thunk for _ in range(num_envs)])
    return envs


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def ppo_parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--policy-epochs", type=int, default=10)
    parser.add_argument("--clip-coeff", type=float, default=0.2)
    parser.add_argument("--ent-coeff", type=float, default=0.0)
    parser.add_argument("--value_coeff", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--icm", default=False, action="store_true")
    parser.add_argument("--icm-lr", type=float, default=3e-4)
    parser.add_argument("--icm-reward-factor", type=float, default=0.05)
    parser.add_argument("--icm-loss-factor", type=float, default=0.2)
    parser.add_argument("--icm-epochs", type=int, default=1)
    parser.add_argument("--log-frequency", type=int, default=500)
    parser.add_argument("--critic-units", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--actor-units", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--icm-units", nargs="+", type=int, default=[512, 512])

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size / args.num_minibatches)
    args.num_updates = int(args.total_timesteps // args.batch_size)

    return args

if __name__ == "__main__":
    main()
