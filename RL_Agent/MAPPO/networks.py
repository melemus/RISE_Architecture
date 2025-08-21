import torch
import torch.nn as nn

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