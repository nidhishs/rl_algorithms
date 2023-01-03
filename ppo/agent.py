import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler

from logger import Logger


class PPOAgent(nn.Module):
    def __init__(self, actor_critic, policy_lr, lr_decay_steps, clip_coeff, value_coeff, entropy_coeff, max_grad_norm, seed):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor_critic = actor_critic
        self.ac_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=policy_lr, eps=1e-5)
        self.ac_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.ac_optimizer, 1, 1e-6, total_iters=lr_decay_steps)
        self.memory = None

        self.clip_coeff = clip_coeff
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

    def act(self, state):
        with torch.no_grad():
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(state)
        return action, log_prob, entropy, value

    def learn(self, global_step, num_mini_batches, policy_epochs, target_kl=None):
        writer = Logger.get().tb

        pi_infos = dict(kl=[], ent=[], cf=[])
        pi_losses, v_losses = [], []

        batch = self.memory.compute_returns_and_advantages()
        
        for epoch in range(policy_epochs):
            batch_indices = BatchSampler(SubsetRandomSampler(range(batch["state"].size(0))), num_mini_batches, False)
            for minibatch_indices in batch_indices:
                minibatch = {k: v[minibatch_indices] for k, v in batch.items()}

                pi_loss, pi_info = self._compute_policy_loss(minibatch)
                v_loss = self._compute_value_loss(minibatch)
                ppo_loss = pi_loss - self.entropy_coeff * pi_info["ent"] + v_loss * self.value_coeff

                self.ac_optimizer.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.ac_optimizer.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                for k, v in pi_info.items():
                    pi_infos[k].append(v.item())

            if target_kl and np.mean(pi_infos["kl"]) > 1.5 * target_kl:
                break

        y_pred, y_true = batch["value"].cpu().numpy(), batch["return"].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.ac_lr_scheduler.step()

        writer.add_scalar("PPOAgent/learning_rate", self.ac_lr_scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("PPOAgent/value_loss", np.mean(v_losses), global_step)
        writer.add_scalar("PPOAgent/policy_loss", np.mean(pi_infos), global_step)
        writer.add_scalar("PPOAgent/entropy", np.mean(pi_infos["ent"]), global_step)
        writer.add_scalar("PPOAgent/approx_kl", np.mean(pi_infos["kl"]), global_step)
        writer.add_scalar("PPOAgent/clip_frac", np.mean(pi_infos["cf"]), global_step)
        writer.add_scalar("PPOAgent/explained_variance", explained_var, global_step)
        Logger.get().info(
            f"{'Agent Optimisation':<20}" + " : "
            + f"Final Loss (pi, v) = ({np.mean(pi_losses):.4f}, {np.mean(v_losses):.4f})"
        )

    def _compute_value_loss(self, mb):
        mb_state, mb_return, mb_value = mb["state"], mb["return"], mb["value"]
        value = self.actor_critic.get_value(mb_state).flatten()

        v_loss_unclipped = (value - mb_return) ** 2
        v_clipped = mb_value + torch.clamp(value - mb_value, -self.clip_coeff, self.clip_coeff)
        v_loss_clipped = (v_clipped - mb_return) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        return v_loss

    def _compute_policy_loss(self, mb):
        mb_state, mb_action, mb_log_prob, mb_advantage = mb["state"], mb["action"], \
            mb["log_prob"], mb["advantage"]
        mb_advantage = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std() + 1e-8)

        _, log_prob, entropy, _ = self.actor_critic.get_action_and_value(mb_state, mb_action)
        log_ratio = log_prob - mb_log_prob
        ratio = log_ratio.exp()

        # Policy loss
        pi_loss_unclipped = -mb_advantage * ratio
        pi_loss_clipped = -mb_advantage * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
        pi_loss = torch.max(pi_loss_unclipped, pi_loss_clipped).mean()

        # Debug Info
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipped = ratio.gt(1 + self.clip_coeff) | ratio.lt(1 - self.clip_coeff)
            clipped_frac = clipped.float().mean()
            ent = entropy.mean()

        return pi_loss, dict(kl=approx_kl, ent=ent, cf=clipped_frac)
