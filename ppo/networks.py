import numpy as np
import torch
import torch.nn as nn

from ppo.core import MaskedCategorical, mlp


def _layer_init(module, stds, bias_const=0.0):
    layers = [l for l in module if (isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d))]
    for i, layer in enumerate(layers):
        torch.nn.init.orthogonal_(layer.weight, stds[i])
        torch.nn.init.constant_(layer.bias, bias_const)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor_units, critic_units, hidden_fn, output_fn):
        """
        Actor Critic network for the agent (continuous action space)

        @param state_dim: observation space dimension tuple
        @param action_dim: action space dimension tuple
        @param actor_units: list of hidden layer sizes for actor network
        @param critic_units: list of hidden layer sizes for critic network
        @param hidden_fn: activation function for all hidden layers
        @param output_fn: activation function for output layer
        """
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.actor_network = mlp([state_dim, *actor_units, action_dim], hidden_fn, output_fn)
        self.critic_network = mlp([state_dim, *critic_units, 1], hidden_fn, output_fn)
        _layer_init(self.actor_network, [np.sqrt(2)] * (len(actor_units)) + [0.01])
        _layer_init(self.critic_network, [np.sqrt(2)] * (len(critic_units)) + [1.0])

    def get_value(self, state):
        return self.critic_network(state)

    def get_action_and_value(self, state, action=None, mask=None):
        raise NotImplementedError


class MLPGaussianActorCritic(ActorCritic):
    def __init__(self, state_dim, action_dim, actor_units, critic_units, hidden_fn=nn.Tanh, output_fn=nn.Identity):
        super().__init__(state_dim, action_dim, actor_units, critic_units, hidden_fn, output_fn)
        self.log_std = nn.Parameter(-0.5 * torch.ones(1, action_dim))

    def get_action_and_value(self, state, action=None, mask=None):
        mu = self.actor_network(state)
        std = torch.exp(self.log_std.expand_as(mu))
        pi = torch.distributions.normal.Normal(mu, std)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action).sum(-1), pi.entropy().sum(-1), self.get_value(state)


class MLPCategoricalActorCritic(ActorCritic):
    def __init__(self, state_dim, action_dim, actor_units, critic_units, hidden_fn=nn.Tanh, output_fn=nn.Identity):
        super().__init__(state_dim, action_dim, actor_units, critic_units, hidden_fn, output_fn)

    def get_action_and_value(self, state, action=None, mask=None):
        logits = self.actor_network(state)
        probs = MaskedCategorical(logits=logits, mask=mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(state)