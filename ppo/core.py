import torch
import torch.nn as nn

class Buffer:
    def __init__(self, max_size, num_envs, state_dim, action_dim, device="cuda"):
        """
        Buffer for storing trajectories experienced by the agent interacting with the environment.
        @param max_size: size of the buffer
        @param num_envs: number of parallel environments
        @param state_dim: dimension of the state space
        @param action_dim: dimension of the action space
        @param device: device to store the buffer on
        """
        super().__init__()
        self.data = {
            "state": torch.zeros((max_size, num_envs, *state_dim)).to(device),
            "action": torch.zeros((max_size, num_envs, *action_dim)).to(device),
            "advantage": torch.zeros((max_size, num_envs)).to(device),
            "reward": torch.zeros((max_size, num_envs)).to(device),
            "return": torch.zeros((max_size, num_envs)).to(device),
            "value": torch.zeros((max_size, num_envs)).to(device),
            "log_prob": torch.zeros((max_size, num_envs)).to(device),
            "termination": torch.zeros((max_size, num_envs)).to(device),
        }
        self.max_size, self.curr_size = max_size, 0
        self.device = device

    def store(self, state, action, reward, termination, value, log_prob):
        assert self.curr_size < self.max_size
        self.data["state"][self.curr_size] = state
        self.data["action"][self.curr_size] = action
        self.data["reward"][self.curr_size] = reward
        self.data["termination"][self.curr_size] = termination
        self.data["value"][self.curr_size] = value
        self.data["log_prob"][self.curr_size] = log_prob
        self.curr_size += 1

    def reset(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.curr_size = 0

    def compute_advantage_and_returns(self, last_value, last_termination, gamma=0.99, gae_lambda=0.95):
        prev_advantage = 0
        for step in reversed(range(self.max_size)):
            if step == self.max_size - 1:
                next_non_terminal = 1.0 - last_termination
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.data["termination"][step + 1]
                next_value = self.data["value"][step + 1]

            delta = self.data["reward"][step] + gamma * next_value * next_non_terminal - self.data["value"][step]
            self.data["advantage"][step] = delta + gamma * gae_lambda * next_non_terminal * prev_advantage
            prev_advantage = self.data["advantage"][step]

        self.data["return"] = self.data["advantage"] + self.data["value"]

    def get(self):
        assert self.curr_size == self.max_size

        # Flatten the data from all the parallel environments (size, num_envs, *) -> (size * num_envs, *)
        return {k: v.flatten(0, 1) for k, v in self.data.items()}


def mlp(dims, hidden_activation=nn.Tanh, output_activation=nn.Identity):
    """
    Construct a multi-layer perceptron with given hidden and output activations

    @param dims: list of dimensions, including input and output dims
    @param hidden_activation: activation function for all hidden layers
    @param output_activation: activation function for output layer
    """

    layers = []
    for i in range(len(dims) - 2):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), hidden_activation()])
    layers.extend([nn.Linear(dims[-2], dims[-1]), output_activation()])
    return nn.Sequential(*layers)
        