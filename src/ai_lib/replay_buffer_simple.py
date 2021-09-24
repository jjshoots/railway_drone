import torch
import numpy as np
from torch.utils.data import Dataset

class ReplayBufferSimple(Dataset):
    """
    Replay buffer that deals with torch tensors but stores as np arrays
    """
    def __init__(self, mem_size):
        self.mem_size = int(mem_size)
        self.counter = 0

        self.state = []
        self.action = []


    def __len__(self):
        return self.mem_size


    def __getitem__(self, idx):
        state = torch.tensor(self.state[idx])
        action = torch.tensor(self.action[idx])

        return state, action


    def push(self, state, action):
        if self.counter == 0:
            self.state = np.zeros((self.mem_size, *state.shape), dtype=np.float32)
            self.action = np.zeros((self.mem_size, *action.shape), dtype=np.float32)

        # i = self.counter % self.mem_size + (int(self.counter / self.mem_size) % self.sparse_forget)
        i = self.counter % self.mem_size
        self.state[i] = state
        self.action[i] = action

        self.counter += 1


    def pull(self, batch_size):
        idx = np.random.randint(min(self.mem_size, self.counter), size=batch_size)

        state = torch.tensor(np.take(self.state, idx, 0))
        action = torch.tensor(np.take(self.action, idx, 0))

        return state, action


    def is_full(self):
        return self.counter >= self.mem_size
