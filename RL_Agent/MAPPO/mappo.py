import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Categorical
import pandas as pd
from latest import NewEnv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # self.actor = nn.Linear(64, action_dim)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # action_probs = torch.softmax(self.actor(x), dim=-1)
        # state_value = self.critic(x)
        # return action_probs, state_value
        mean = self.mean(x)
        std = self.log_std.exp()
        state_value = self.critic(x)
        return mean, std, state_value

class MultiAgentPPO:
    def __init__(self, n_agents, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=0.2):
        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.agents = [ActorCritic(state_dim, action_dim) for _ in range(n_agents)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=lr) for agent in self.agents]
    
    def choose_action(self, agent, state):
        state = torch.FloatTensor(state)
        # action_probs, _ = self.agents[agent](state)
        # dist = Categorical(action_probs)

        mean, std, _ = self.agents[agent](state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # return action.item(), dist.log_prob(action)
        return action.detach().numpy(), log_prob
    
    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae * (1 - dones[t])
            advantages.insert(0, gae)
        return advantages

    
    def update(self, trajectories):
        for agent_idx in range(self.n_agents):
            states, actions, log_probs, rewards, dones, values = zip(*trajectories[agent_idx])
            
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            # actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(log_probs)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            values = torch.FloatTensor(values)
            
            advantages = torch.FloatTensor(self.compute_advantages(rewards, values, dones))
            # targets = advantages + values[:-1]
            targets = advantages + values


            for _ in range(10):  
                # action_probs, state_values = self.agents[agent_idx](states)
                # dist = Categorical(action_probs)
                # new_log_probs = dist.log_prob(actions)
                mean, std, state_values = self.agents[agent_idx](states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)

                # Actor Loss (Clipped Surrogate Objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                # Critic Loss
                critic_loss = nn.MSELoss()(state_values.squeeze(), targets)

                # Combined Loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()

                # Update Parameters
                self.optimizers[agent_idx].zero_grad()
                loss.backward()
                self.optimizers[agent_idx].step()

def train_multi_agent_ppo(env, n_agents, state_dim, action_dim, episodes=500, steps_per_episode=200):
    agent = MultiAgentPPO(n_agents, state_dim, action_dim)
    episode_rewards = []
    reward_1_history,reward_2_history,reward_3_history = [],[],[]
    done_1_history,done_2_history,done_3_history = [],[],[]

    for episode in range(episodes):
        trajectories = [[] for _ in range(n_agents)]
        states = env.reset()
        total_reward = 0
        reward1,reward2,reward3 = 0,0,0
        done1,done2,done3 = 0,0,0
        
        for _ in range(steps_per_episode):
            actions, log_probs, values = [], [], []
            for i in range(n_agents):
                action, log_prob = agent.choose_action(i, states[i])
                actions.append(action)
                log_probs.append(log_prob)
                # _, value = agent.agents[i](torch.FloatTensor(states[i]))
                _, _, value = agent.agents[i](torch.FloatTensor(states[i]))
                values.append(value.item())
            
            next_states, rewards, dones, _ = env.step(actions)

            reward1+=rewards[0]
            reward2+=rewards[1]
            reward3+=rewards[2]

            if dones[0]:
                done1+=1
            if dones[1]:
                done2+=1
            if dones[2]:
                done3+=1

            for i in range(n_agents):
                trajectories[i].append((states[i], actions[i], log_probs[i], rewards[i], dones[i], values[i]))
                total_reward += rewards[i]

            
            states = next_states
            if all(dones):
                break
        
        episode_rewards.append(total_reward)
        reward_1_history.append(reward1)
        reward_2_history.append(reward2)
        reward_3_history.append(reward3)
        # Update policy and value networks

        done_1_history.append(done1)
        done_2_history.append(done2)
        done_3_history.append(done3)

        agent.update(trajectories)

        print(f"Episode {episode + 1}: Done")


    print("Done 1 history: ",done_1_history)
    print("Done 2 history: ",done_2_history)
    print("Done 3 history: ",done_3_history)
    
    max_reward,max_reward1,max_reward2,max_reward3 = max(episode_rewards),max(reward_1_history),max(reward_2_history),max(reward_3_history)
    incremental_score_history = [max_reward-i for i in episode_rewards]
    incremental_reward1_history = [max_reward1-i for i in reward_1_history]
    incremental_reward2_history = [max_reward2-i for i in reward_2_history]
    incremental_reward3_history = [max_reward3-i for i in reward_3_history]



    plt.figure()
    plt.plot(incremental_reward1_history, label='Agent 1')
    plt.plot(incremental_reward2_history, label='Agent 2')
    plt.plot(incremental_reward3_history, label='Agent 3')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.rcParams.update({'font.size': 12})
    plt.title('Rewards for Each Agent')
    plt.legend()
    plt.savefig('Plots/rewards_plot.png') 

    plt.figure()
    plt.plot(incremental_score_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.rcParams.update({'font.size': 12})
    plt.title('Total Score per Episode')
    plt.savefig('Plots/total_score_plot.png')

    

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
    n_agents = 3  
    state_dim = 5  
    action_dim = 4 

    train_multi_agent_ppo(env, n_agents, state_dim, action_dim)
