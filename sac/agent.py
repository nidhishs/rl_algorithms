import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sac.networks import MLPActor, MLPCritic
from logger import Logger

class SACAgent(nn.Module):
    def __init__(
        self, state_dim, action_dim, actor_units=[256, 256], critic_units=[256, 256], gamma=0.99, tau=0.005, q_lr=1e-3, pi_lr=3e-4,
        log_std_min=-5, log_std_max=2, actor_update_interval=2, target_update_interval=1, seed=0
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = MLPActor(np.prod(state_dim), np.prod(action_dim), actor_units, log_std_min, log_std_max)
        self.critic_1 = MLPCritic(np.prod(state_dim), np.prod(action_dim), critic_units)
        self.critic_2 = MLPCritic(np.prod(state_dim), np.prod(action_dim), critic_units)
        self.target_critic_1 = MLPCritic(np.prod(state_dim), np.prod(action_dim), critic_units)
        self.target_critic_2 = MLPCritic(np.prod(state_dim), np.prod(action_dim), critic_units)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.q_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=q_lr)
        self.pi_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.target_entropy = -np.prod(action_dim)
        self.log_alpha = nn.Parameter(torch.tensor([0.0]))
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=pi_lr)

        self.gamma, self.tau = gamma, tau
        self.actor_update_interval, self.target_update_interval = actor_update_interval, target_update_interval

    def act(self, state):
        with torch.no_grad():
            action, log_prob = self.actor(state)
        return action

    def optimize(self, batch, global_step):
        writer = Logger.get_logger().tb

        q_loss, q_info = self._compute_q_loss(batch)

        pi_loss, alpha_loss = None, None
        if global_step % self.actor_update_interval == 0:
            for _ in range(self.actor_update_interval):
                pi_loss = self._compute_pi_loss(batch["state"])
                alpha_loss = self._tune_entropy_coeff(batch["state"])
        
        if global_step % self.target_update_interval == 0:
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
        
        
        writer.add_scalar("SACAgent/critic_1_value", q_info["q1"].mean().item(), global_step)
        writer.add_scalar("SACAgent/critic_2_value", q_info["q2"].mean().item(), global_step)
        writer.add_scalar("SACAgent/critic_1_loss", q_info["q1_loss"].item(), global_step)
        writer.add_scalar("SACAgent/critic_2_loss", q_info["q2_loss"].item(), global_step)
        writer.add_scalar("SACAgent/critic_loss", q_loss.item(), global_step)
        if pi_loss:
            writer.add_scalar("SACAgent/actor_loss", pi_loss.item(), global_step)
        if alpha_loss:
            writer.add_scalar("SACAgent/alpha_loss", alpha_loss.item(), global_step)

    def _compute_pi_loss(self, b_state):
        action, log_prob = self.actor(b_state)
        q1_values = self.critic_1(b_state, action)
        q2_values = self.critic_2(b_state, action)
        min_q_values = torch.min(q1_values, q2_values).flatten()
        pi_loss = (self.alpha * log_prob - min_q_values).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        return pi_loss

    def _compute_q_loss(self, batch):
        b_state, b_action, b_next_state, b_reward, b_termination = \
            batch["state"], batch["action"], batch["next_state"], batch["reward"], batch["termination"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor(b_next_state)
            q1_next_target = self.target_critic_1(b_next_state, next_action)
            q2_next_target = self.target_critic_2(b_next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = b_reward.flatten() + (1 - b_termination.flatten()) * self.gamma * min_q_next_target.flatten()
        
        q1_values = self.critic_1(b_state, b_action).flatten()
        q2_values = self.critic_2(b_state, b_action).flatten()
        q1_loss = F.mse_loss(q1_values, next_q_value)
        q2_loss = F.mse_loss(q2_values, next_q_value)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q_loss, dict(q1=q1_values, q1_loss=q1_loss, q2=q2_values, q2_loss=q2_loss)

    def _tune_entropy_coeff(self, b_state):
        with torch.no_grad():
            _, log_prob = self.actor(b_state)
        alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        return alpha_loss

