# Safe Reinforcement Learning Agent (CPPO/Dreamer)
"""
Safe Actor-Critic Agent for AutoGuard-RL
Includes safety penalties based on CLIP-based risk assessment.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SafeAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, safety_lambda=10.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.critic = ValueNetwork(obs_dim).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.safety_lambda = safety_lambda

    def select_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(obs)
        return action.cpu().numpy()

    def update(self, obs, actions, returns, safety_costs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        safety_costs = torch.tensor(safety_costs, dtype=torch.float32).to(self.device)

        pred_actions = self.actor(obs)
        pred_values = self.critic(obs)
        value_loss = (pred_values - returns).pow(2).mean()
        actor_loss = (pred_actions - actions).pow(2).mean()
        safety_penalty = self.safety_lambda * safety_costs.mean()

        total_loss = actor_loss + 0.5 * value_loss + safety_penalty
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "safety_penalty": safety_penalty.item(),
            "total_loss": total_loss.item()
        }
