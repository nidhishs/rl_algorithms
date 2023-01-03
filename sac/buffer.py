import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, size, state_dim, action_dim, num_envs=1, device=None, seed=0):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = max(size//num_envs, 1)
        self.data = {
            "state": torch.zeros((size, num_envs, *state_dim)).to(self.device),
            "action": torch.zeros((size, num_envs, *action_dim)).to(self.device),
            "reward": torch.zeros((size, num_envs)).to(self.device),
            "next_state": torch.zeros((size, num_envs, *state_dim)).to(self.device),
            "termination": torch.zeros((size, num_envs)).to(self.device),
        }

        self.max_size, self.curr_size, self.is_full = size, 0, False
        self._rng = np.random.default_rng(seed=seed)

    def add(self, state, action, reward, next_state, termination):
        self.data["state"][self.curr_size] = state
        self.data["action"][self.curr_size] = action
        self.data["reward"][self.curr_size] = reward
        self.data["next_state"][self.curr_size] = next_state
        self.data["termination"][self.curr_size] = termination
        self.curr_size += 1
        if self.curr_size == self.max_size:
            self.is_full = True
            self.curr_size = 0
    
    def sample(self, batch_size):
        upper_bound = self.max_size if self.is_full else self.curr_size
        indices = self._rng.choice(upper_bound, size=batch_size)
        # Flatten the data from all the parallel environments (size, num_envs, *) -> (size * num_envs, *)
        return {k: v.flatten(0, 1)[indices] for k, v in self.data.items()}