import torch
import torch.nn as nn
import torch.optim as optim
from networks import ActorCritic

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