import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    """Gaussian actor for continuous control. Outputs mean and log_std per actuator."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
        )
        self.mean_head    = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x       = self.net(state)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_dist(self, state):
        mean, log_std = self(state)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, state):
        dist     = self.get_dist(state)
        action   = dist.rsample()                       # reparameterised
        log_prob = dist.log_prob(action).sum(-1)        # sum over action dims
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, state):
        return self.net(state)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor  = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def forward(self, state):
        action, log_prob = self.actor.sample(state)
        value = self.critic(state)
        return action, log_prob, value


class RunningMeanStd:
    """Tracks running mean and variance for online normalization."""
    def __init__(self, shape):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_count / total
        self.var   = (self.var * self.count + batch_var * batch_count +
                      delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.policy    = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.obs_rms   = RunningMeanStd(shape=(state_dim,))

        self.gamma      = 0.99
        self.lam        = 0.95
        self.clip_eps   = 0.2
        self.epochs     = 10
        self.batch_size = 64

    def compute_gae(self, rewards, values, dones):
        """Generalised Advantage Estimation."""
        advantages = []
        gae    = 0
        values = values + [0.0]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae   = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single observation using the running statistics."""
        return self.obs_rms.normalize(obs).astype(np.float32)

    def update(self, states, actions, log_probs_old, rewards, dones, values):
        # Update running stats with all raw states collected this rollout
        raw = np.stack([s.cpu().numpy() for s in states])
        self.obs_rms.update(raw)

        vals_scalar = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in values]

        advantages  = self.compute_gae(rewards, vals_scalar, dones)
        returns_lst = [adv + val for adv, val in zip(advantages, vals_scalar)]

        # Detach everything — these are fixed reference data from the old policy
        states        = torch.stack(states).detach().to(device)           # [T, state_dim]
        actions       = torch.stack(actions).detach().to(device)          # [T, action_dim]
        log_probs_old = torch.stack(log_probs_old).detach().to(device)    # [T]
        returns       = torch.tensor(returns_lst,  dtype=torch.float32, device=device)
        advantages    = torch.tensor(advantages,   dtype=torch.float32, device=device)
        advantages    = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(states)
        for _ in range(self.epochs):
            idx = torch.randperm(T, device=device)
            for start in range(0, T, self.batch_size):
                b = idx[start: start + self.batch_size]

                b_states     = states[b]
                b_actions    = actions[b]
                b_log_old    = log_probs_old[b]
                b_returns    = returns[b]
                b_advantages = advantages[b]

                # Re-evaluate under current policy
                dist          = self.policy.actor.get_dist(b_states)
                log_probs_new = dist.log_prob(b_actions).sum(-1)
                entropy       = dist.entropy().sum(-1).mean()
                state_values  = self.policy.critic(b_states).squeeze(-1)

                ratios = torch.exp(log_probs_new - b_log_old)
                surr1  = ratios * b_advantages
                surr2  = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values, b_returns)
                loss        = actor_loss + 0.5 * critic_loss - 0.02 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            self.policy.load_state_dict(ckpt["policy"])
            if "obs_rms" in ckpt:
                self.obs_rms.load_state_dict(ckpt["obs_rms"])
        else:
            # legacy checkpoint (policy state_dict only)
            self.policy.load_state_dict(ckpt)

