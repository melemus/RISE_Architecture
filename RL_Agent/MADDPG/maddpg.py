import torch as T
import torch.nn.functional as F
import numpy as np
import random
from agent import Agent

random.seed(42)
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        
        random.seed(42)

        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            # print("agent_idx" , agent_idx)

            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions[agent_idx], n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))
            # self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
            #                 n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
            #                 chkpt_dir=chkpt_dir))



    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        combined_action =  np.zeros(self.n_actions)
        random_action =  []
        for agent_idx, agent in enumerate(self.agents):
            
            action = agent.choose_action_agent(raw_obs[agent_idx])
            # combined_action += action
            actions.append(action)
        # random_action = random.choice(actions)
        # combined_action/=self.n_actions
        # combined_action = [actions[0][0]+actions[1][0]+actions[2][0],actions[0][1]+actions[1][1]+actions[2][1],actions[0][2]+actions[1][2]+actions[2][2]]
        # combined_action = [sum(a)/self.n_agents for a in zip(*actions)]
        
        # print(f'Random_action: {random_action}')
        # print(f'combined_action: {combined_action}')
        # print('--------------------')
        # print(f'actions: {actions}')
        # print('--------------------')
        # print(combined_action)
        # print(actions)
        # print('------------')
        # print(combined_action)
        # return random_action
        # return combined_action
        return actions
    
    T.autograd.set_detect_anomaly(True)
    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states1 = T.tensor(states, dtype=T.float).to(device)
        # print(f'actions: {actions}')
        actions1 = T.tensor(actions, dtype=T.float).to(device)
        # actions1 = T.tensor(np.column_stack(actions), dtype=T.float).to(device)
        
        rewards1 = T.tensor(rewards,dtype=T.float).to(device)
        states_1 = T.tensor(states_, dtype=T.float).to(device)
        dones1 = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            
            new_pi = agent.target_actor.forward(new_states)
            

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions1[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)


        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_1, new_actions).flatten()
            critic_value_[dones1[:,0]] = 0.0
            critic_value = agent.critic.forward(states1, old_actions).flatten()

            target = rewards1[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            # critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states1, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            # actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()