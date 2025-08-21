import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from latest import NewEnv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Multi-Agent A2C Class
class MultiAgentA2C:
    def __init__(self, n_agents, state_dim, action_dim, lr=1e-4, gamma=0.99):
        self.n_agents = n_agents
        self.actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critic = Critic(state_dim)
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma

    def choose_actions(self, states):
        actions = []
        log_probs = []
        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i])
            action_probs = actor(state)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
            actions.append(action)
            log_probs.append(log_prob)
        return actions, log_probs

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Calculate values and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = targets - values

        # Update Critic
        critic_loss = advantages.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actors
        for i, actor in enumerate(self.actors):
            action_probs = actor(states[i])
            log_prob = torch.log(action_probs[actions[i]])
            actor_loss = -(log_prob * advantages[i].detach())
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

# Training Loop
def train_multi_agent_a2c(env, n_agents, state_dim, action_dim, episodes=500, steps_per_episode=200):
    agent = MultiAgentA2C(n_agents, state_dim, action_dim)
    reward_history = []
    for episode in range(episodes):
        states = env.reset()
        total_rewards = [0] * n_agents

        for _ in range(steps_per_episode):
            actions, log_probs = agent.choose_actions(states)
            print(actions)
            next_states, rewards, dones, _ = env.step(actions)
            agent.update(states, actions, rewards, next_states, dones)
            states = next_states
            total_rewards = [r + reward for r, reward in zip(total_rewards, rewards)]
            if all(dones):
                break
        episode_total_reward = sum(total_rewards)
        reward_history.append(episode_total_reward)

        print(f"Episode {episode + 1}, Total Rewards: {total_rewards}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), reward_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

dfOrig = pd.read_excel('Data/FinalData.xlsx')
df = dfOrig.copy()
scaler = MinMaxScaler()
df[['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Zoom','Focus','Contrast']] = scaler.fit_transform(df[['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Zoom','Focus','Contrast']])


v= {"ol": [0.5*min(df['Orientation_Loss']),1.5*max(df['Orientation_Loss'])],
        "ec": [0.5*min(df['Edge_Coverage']),1.5*max(df['Edge_Coverage'])],
        "at": [0.5*min(df['Average_Thickness']),1.5*max(df['Average_Thickness'])],
        "as": [0.5*min(df['Average_Separation']),1.5*max(df['Average_Separation'])],
        "de": [0.5*min(df['Distance_Entropy']),1.5*max(df['Distance_Entropy'])],
        "z": [0.5*min(df['Zoom']),1.5*max(df['Zoom'])],
        "f": [0.5*min(df['Focus']),1.5*max(df['Focus'])],
        "c": [0.5*min(df['Contrast']),1.5*max(df['Contrast'])]
        }
corr_vals = [list(df.corr()['Zoom'])[:5],list(df.corr()['Focus'])[:5],list(df.corr()['Contrast'])[:5]]


if __name__ == "__main__":
    env = NewEnv(df, corr_vals)
    n_agents = 3  # ( zoom, focus, contrast)
    state_dim = 5  
    action_dim = 4  

    train_multi_agent_a2c(env, n_agents, state_dim, action_dim)
