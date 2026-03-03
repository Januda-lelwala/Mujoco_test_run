import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda")

class ActorCritic:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim) 

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value
    
class PPO:
    def __init__(self,state_dim,action_dim):
        self.policy=ActorCritic(state_dim,action_dim).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=3e-4)

        self.gamma=0.99
        self.lam=0.95
        self.clip_eps=0.2
        self.epochs=10
        self.batch_size=64

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.policy.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.policy.critic(state).squeeze()
        return action.item(), log_prob.item(), value.item()

    def update(self, states, actions, log_probs_old, rewards, dones, values):
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Convert to tensors
        states      = torch.FloatTensor(states).to(device)
        actions     = torch.LongTensor(actions).to(device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(device)
        returns     = torch.FloatTensor(returns).to(device)
        advantages  = torch.FloatTensor(advantages).to(device)
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        for _ in range(self.epochs):
            # Shuffle indices for mini-batch updates
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]

                b_states     = states[batch_idx]
                b_actions    = actions[batch_idx]
                b_log_probs_old = log_probs_old[batch_idx]
                b_returns    = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                # Current policy
                action_probs = self.policy.actor(b_states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                state_values = self.policy.critic(b_states).squeeze()

                # PPO clipped surrogate objective
                ratios = torch.exp(log_probs_new - b_log_probs_old)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = F.mse_loss(state_values, b_returns)

                # Total loss (entropy bonus encourages exploration)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

