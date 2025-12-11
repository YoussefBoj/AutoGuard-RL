# Dreamer-style World Model (RSSM)
"""
RSSM World Model — Predicts latent dynamics from visual observations.
Used by AutoGuard-RL to imagine trajectories safely.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 10 * 10, feature_dim)  # adjust based on image size

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, feature_dim=128, output_shape=(3, 84, 84)):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 128 * 10 * 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, output_shape[0], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 10, 10)
        return self.decoder(x)


class RSSM(nn.Module):
    """Simplified Dreamer Recurrent State-Space Model."""
    def __init__(self, obs_dim=128, action_dim=2, hidden_dim=256, state_dim=128):
        super().__init__()
        self.gru = nn.GRU(obs_dim + action_dim, hidden_dim, batch_first=True)
        self.to_state = nn.Linear(hidden_dim, state_dim)
        self.to_obs = nn.Linear(state_dim, obs_dim)

    def forward(self, obs_seq, action_seq, hidden=None):
        x = torch.cat([obs_seq, action_seq], dim=-1)
        out, h = self.gru(x, hidden)
        state = self.to_state(out)
        pred_obs = self.to_obs(state)
        return state, pred_obs


class WorldModel(nn.Module):
    """Full world model = Encoder + RSSM + Decoder."""
    def __init__(self, image_shape=(3, 84, 84), action_dim=2):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=image_shape[0])
        self.rssm = RSSM(obs_dim=128, action_dim=action_dim)
        self.decoder = ConvDecoder()

    def forward(self, obs_imgs, actions):
        obs_emb = self.encoder(obs_imgs)
        obs_seq = obs_emb.unsqueeze(1)
        act_seq = actions.unsqueeze(1)
        state, pred = self.rssm(obs_seq, act_seq)
        recon = self.decoder(pred[:, -1])
        return state, recon
