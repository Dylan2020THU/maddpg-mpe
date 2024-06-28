import torch
import torch.nn as nn
# import torch.nn.functional as F
from .transformer import Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Define the actor network with transformer
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.embedding_dim = 64
        self.transformer = Transformer(num_layers=3, hidden_size=self.embedding_dim, num_heads=4, dropout=0.1)
        self.positional_encoding = PositionalEncoding(self.embedding_dim)
        self.fc_in = nn.Linear(args.obs_shape[agent_id], self.embedding_dim)
        self.action_out = nn.Linear(self.embedding_dim, args.action_shape[agent_id])

    def forward(self, x):
        x = self.fc_in(x).unsqueeze(0)  # Add sequence dimension (seq_len=1)
        x = self.positional_encoding(x)
        x = self.transformer(x, None).squeeze(0)  # Remove sequence dimension
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

# Define the critic network with transformer
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.embedding_dim = 64
        self.transformer = Transformer(num_layers=3, hidden_size=self.embedding_dim, num_heads=4, dropout=0.1)
        self.positional_encoding = PositionalEncoding(self.embedding_dim)
        self.fc_in = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), self.embedding_dim)
        self.q_out = nn.Linear(self.embedding_dim, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = self.fc_in(x).unsqueeze(0)  # Add sequence dimension (seq_len=1)
        x = self.positional_encoding(x)
        x = self.transformer(x, None).squeeze(0)  # Remove sequence dimension
        q_value = self.q_out(x)
        return q_value
