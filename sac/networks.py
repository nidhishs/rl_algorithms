import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal


def mlp(dims, hidden_activation=nn.Tanh, output_activation=nn.Identity):
    """
    Construct a multi-layer perceptron with given hidden and output activations.
    @param dims: list of dimensions, including input and output dims
    @param hidden_activation: activation function for all hidden layers
    @param output_activation: activation function for output layer
    """
    layers = []
    for i in range(len(dims) - 2):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), hidden_activation()])
    layers.extend([nn.Linear(dims[-2], dims[-1]), output_activation()])
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_units=[256, 256], log_std_min=-5, log_std_max=2):
        """
        Actor network for the agent.
        @param state_dim: dimension of the state space
        @param action_dim: dimension of the action space
        @param actor_units: list of hidden layer sizes for actor network
        @param init_w: Limits of uniform distribution for mean and log_std networks.
        @param log_std_min: Minimum value for log_std.
        @param log_std_max: Maximum value for log_std.
        @param seed: Random seed.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc = mlp([state_dim] + actor_units, hidden_activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu = nn.Linear(actor_units[-1], action_dim)
        self.log_std_linear = nn.Linear(actor_units[-1], action_dim)

    def _get_distribution_params(self, state):
        x = self.fc(state)
        mu = self.mu(x)
        log_std = torch.tanh(self.log_std_linear(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        return mu, log_std
    
    def forward(self, state, epsilon=1e-6):
        """
        Actor network returns action and log probability of the action for a given state.
        @param state: state of the environment, shape = (batch_size, state_dim)
        @param epsilon: small value to avoid log(0)
        @return: action, shape = (batch_size, action_dim) and log prob of action, shape=(batch_size, 1) 
        """
        mu, log_std = self._get_distribution_params(state)
        std = log_std.exp()
        
        e = Normal(0, 1).sample()
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob.sum(1, keepdim=True)

class MLPCritic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_units=[256, 256]):
        """
        Critic network for the agent.
        @param state_dim: dimension of the state space
        @param action_dim: dimension of the action space
        @param critic_units: list of hidden layer sizes for critic network
        @param seed: Random seed.
        """
        super().__init__()
        self.fc = mlp([state_dim + action_dim] + critic_units + [1], hidden_activation=nn.ReLU)

    def forward(self, state, action):
        """
        Value network maps (state, action) pairs to Q-values.
        @param state: state tensor, shape = (batch_size, state_dim)
        @param action: action tensor, shape = (batch_size, action_dim)
        @return: Q-value tensor, shape = (batch_size, 1)
        """
        x = torch.cat((state, action), dim=1)
        return self.fc(x)
