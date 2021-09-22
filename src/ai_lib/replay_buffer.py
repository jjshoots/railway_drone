import torch
import numpy as np

class ReplayBuffer():
    """
    Replay buffer that deals with torch tensors but stores as np arrays
    """
    def __init__(self, mem_size, sparse_forget=4):
        self.mem_size = int(mem_size)
        self.sparse_forget = sparse_forget
        self.counter = 0

        self.state = []
        self.next_state = []
        self.action = []
        self.next_action = []
        self.reward = []
        self.done = []

    def push(self, state, next_state, action, next_action, reward, done):
        if self.counter == 0:
            self.state = np.zeros((self.mem_size, *state.shape), dtype=np.float32)
            self.next_state = np.zeros((self.mem_size, *next_state.shape), dtype=np.float32)
            self.action = np.zeros((self.mem_size, *action.shape), dtype=np.float32)
            self.next_action = np.zeros((self.mem_size, *next_action.shape), dtype=np.float32)
            self.reward = np.zeros((self.mem_size, *reward.shape), dtype=np.float32)
            self.done = np.zeros((self.mem_size, *done.shape), dtype=np.float32)

        # i = self.counter % self.mem_size + (int(self.counter / self.mem_size) % self.sparse_forget)
        i = self.counter % self.mem_size
        self.state[i] = state.detach().cpu().numpy()
        self.next_state[i] = next_state.detach().cpu().numpy()
        self.action[i] = action.detach().cpu().numpy()
        self.next_action[i] = next_action.detach().cpu().numpy()
        self.reward[i] = reward.detach().cpu().numpy()
        self.done[i] = done.detach().cpu().numpy()

        self.counter += 1


    def pull(self, batch_size, device='cuda:0'):
        idx = np.random.randint(min(self.mem_size, self.counter), size=batch_size)

        state = torch.tensor(np.take(self.state, idx, 0)).to(device)
        next_state = torch.tensor(np.take(self.next_state, idx, 0)).to(device)
        action = torch.tensor(np.take(self.action, idx, 0)).to(device)
        next_action = torch.tensor(np.take(self.next_action, idx, 0)).to(device)
        reward = torch.tensor(np.take(self.reward, idx, 0)).to(device)
        done = torch.tensor(np.take(self.done, idx, 0)).to(device)

        return state, next_state, action, next_action, reward, done


    def is_full(self):
        return len(self.state) >= self.mem_size
