import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create a multi-layer perceptron (MLP)
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        # Use the specified activation for intermediate layers
        # and output_activation for the final layer
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

# Actor network for generating actions
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # MLP to map observations to actions
        self.net = mlp([obs_dim] + hidden_sizes + [act_dim], activation, nn.Tanh)

    def forward(self, obs):
        # Generate action from observations
        return self.net(obs)

# Critic network for evaluating state-action pairs (Q-values)
class Critic(nn.Module):
    def __init__(self, obs_dims, act_dims, hidden_sizes, activation):
        super().__init__()
        self.obs_dims = obs_dims  # Dimensions of observations for all agents
        self.act_dims = act_dims  # Dimensions of actions for all agents
        input_dim = sum(obs_dims) + sum(act_dims)  # Total input size (obs + act)
        self.net = mlp([input_dim] + hidden_sizes + [1], activation)

    def forward(self, obs_list, act_list):
        # Concatenate observations and actions from all agents
        obs_combined = torch.cat([obs_list[i].view(-1, self.obs_dims[i]) for i in range(len(self.obs_dims))], dim=-1)
        act_combined = torch.cat([act_list[i].view(-1, self.act_dims[i]) for i in range(len(self.act_dims))], dim=-1)
        combined = torch.cat([obs_combined, act_combined], dim=-1)
        q = self.net(combined)  # Compute Q-value
        return torch.squeeze(q, -1)

# Agent class that contains an actor and interacts with the environment
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, lr=3e-4, load_policy=None):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.critic = None  # Critic is shared across agents
        self.target_actor = deepcopy(self.actor).to(device)  # Target network for stability

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        if load_policy is not None:
            self.load_weights(load_policy + '.pt')

    def act(self, obs):
        # Generate action based on the current policy
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return self.actor(obs).cpu().data.numpy()

    def update_actor(self, shared_critic, replay_buffer, agent_idx, gamma, polyak):
        # Sample a batch from the replay buffer
        obs, act, rew, next_obs, done = replay_buffer.sample()
        obs = [torch.as_tensor(o, dtype=torch.float32, device=device) for o in obs]
        act = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in act]
        rew = torch.as_tensor(rew, dtype=torch.float32, device=device)
        next_obs = [torch.as_tensor(no, dtype=torch.float32, device=device) for no in next_obs]
        done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # Update actor by maximizing Q-value
        obs_i = obs[agent_idx]
        act_i = self.actor(obs_i)  # Get current agent's actions
        q_val = shared_critic.critic(obs, [act if i != agent_idx else act_i for i, act in enumerate(act)])
        actor_loss = -q_val.mean()  # Minimize negative Q-value

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update for target actor
        with torch.no_grad():
            for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def save_weights(self, filename):
        """Save the model's weights."""
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        """Load weights from a file."""
        self.load_state_dict(torch.load(filename))
        print("Loaded checkpoint")

# Shared critic class used by all agents
class SharedCritic:
    def __init__(self, obs_dims, act_dims, hidden_sizes, activation, lr=1e-3):
        self.critic = Critic(obs_dims, act_dims, hidden_sizes, activation).to(device)
        self.target_critic = deepcopy(self.critic).to(device)  # Target network for stability
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

    def update(self, replay_buffer, gamma, polyak, agents):
        # Sample a batch from the replay buffer
        obs, act, rew, next_obs, done = replay_buffer.sample()
        obs = [torch.as_tensor(o, dtype=torch.float32, device=device) for o in obs]
        act = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in act]
        rew = torch.as_tensor(rew, dtype=torch.float32, device=device)
        next_obs = [torch.as_tensor(no, dtype=torch.float32, device=device) for no in next_obs]
        done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # Compute target Q-value
        with torch.no_grad():
            next_act = [agent.target_actor(next_obs[i]) for i, agent in enumerate(agents)]
            target_q = self.target_critic(next_obs, next_act)
            target_q = rew + gamma * (1 - done) * target_q

        # Compute critic loss
        current_q = self.critic(obs, act)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft update for target critic
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

# Replay buffer for storing and sampling experiences
class ReplayBuffer:
    def __init__(self, obs_dims, act_dims, size):
        # Initialize buffers for observations, actions, rewards, etc.
        self.obs_buf = [np.zeros((size, dim), dtype=np.float32) for dim in obs_dims]
        self.act_buf = [np.zeros((size, dim), dtype=np.float32) for dim in act_dims]
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = [np.zeros((size, dim), dtype=np.float32) for dim in obs_dims]
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # Store a new experience in the buffer
        for i, (o, n, a) in enumerate(zip(obs, next_obs, act)):
            self.obs_buf[i][self.ptr] = o
            self.next_obs_buf[i][self.ptr] = n
            self.act_buf[i][self.ptr] = a
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # Circular buffer
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=512):
        # Sample a random batch of experiences from the buffer
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            [buf[idxs] for buf in self.obs_buf],
            [buf[idxs] for buf in self.act_buf],
            self.rew_buf[idxs],
            [buf[idxs] for buf in self.next_obs_buf],
            self.done_buf[idxs],
        )
