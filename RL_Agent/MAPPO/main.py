import torch
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Categorical
import pandas as pd
from latest import NewEnv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from agents import MultiAgentPPO
import time

def train_multi_agent_ppo(env, n_agents, state_dim, action_dim, episodes=200, steps_per_episode=50):
    agent = MultiAgentPPO(n_agents, state_dim, action_dim)
    episode_rewards = []
    reward_1_history,reward_2_history,reward_3_history = [],[],[]

    for episode in range(episodes):
        trajectories = [[] for _ in range(n_agents)]
        states = env.reset()
        total_reward = 0
        reward1,reward2,reward3 = 0,0,0
        
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


        agent.update(trajectories)

        print(f"Episode {episode + 1}: Done")

    
    max_reward,max_reward1,max_reward2,max_reward3 = max(episode_rewards),max(reward_1_history),max(reward_2_history),max(reward_3_history)
    incremental_score_history = [max_reward-i for i in episode_rewards]
    incremental_reward1_history = [max_reward1-i for i in reward_1_history]
    incremental_reward2_history = [max_reward2-i for i in reward_2_history]
    incremental_reward3_history = [max_reward3-i for i in reward_3_history]

    return incremental_score_history,[incremental_reward1_history,incremental_reward2_history,incremental_reward3_history]


def test_multi_agent_ppo(env, agent, n_agents, steps_per_episode=80, episodes=200):
    episode_rewards = []
    reward_1_history, reward_2_history, reward_3_history = [], [], []
    
    for episode in range(episodes):
        states = env.reset()
        total_reward = 0
        reward1, reward2, reward3 = 0, 0, 0
        
        for _ in range(steps_per_episode):
            actions = []
            for i in range(n_agents):
                action, _ = agent.choose_action(i, states[i])
                actions.append(action)

            next_states, rewards, dones, _ = env.step(actions)

            reward1 += rewards[0]
            reward2 += rewards[1]
            reward3 += rewards[2]
            total_reward += sum(rewards)
            states = next_states

            if all(dones):
                break

        episode_rewards.append(total_reward)
        reward_1_history.append(reward1)
        reward_2_history.append(reward2)
        reward_3_history.append(reward3)
        print(f"Test Episode {episode + 1}: Total Reward {total_reward}")

    max_reward,max_reward1,max_reward2,max_reward3 = max(episode_rewards),max(reward_1_history),max(reward_2_history),max(reward_3_history)
    incremental_score_history = [max_reward-i for i in episode_rewards]
    incremental_reward1_history = [max_reward1-i for i in reward_1_history]
    incremental_reward2_history = [max_reward2-i for i in reward_2_history]
    incremental_reward3_history = [max_reward3-i for i in reward_3_history]
    
    return incremental_score_history,[incremental_reward1_history,incremental_reward2_history,incremental_reward3_history]



if __name__ == "__main__":
    startTime = time.time()
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

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    # Training Environment
    print("Training Started...")
    train_env = NewEnv(train_df, corr_vals)
    n_agents = 3
    state_dim = 5
    action_dim = 4
    agent = MultiAgentPPO(n_agents, state_dim, action_dim)

    train_scores, train_rewards = train_multi_agent_ppo(train_env, n_agents, state_dim, action_dim, episodes=500)
    print("Max Training Reward: ",max(train_scores))
    print("Training Completed!")

    # Testing Environment
    print("Testing Started...")
    test_env = NewEnv(test_df, corr_vals)
    test_scores, test_rewards = test_multi_agent_ppo(test_env, agent, n_agents)
    print("Max Testing Rewards: ",max(test_scores))
    print("Testing Completed!")

    print('It took', time.time()-startTime, 'seconds.')

    #Put train and test in csv
    
    # Plot Training Results
    plt.figure()
    plt.plot(train_scores, label='Training Scores', linestyle='solid', color='black')
    plt.plot(test_scores, label='Testing Scores', linestyle=':', color='black')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training and Testing Performance', fontsize=12)
    plt.legend(loc='center right', fontsize=10)
    plt.grid(False)
    plt.savefig('Plots/train_test_rewards.png')
    plt.show()
    # plt.figure()
    # plt.plot(train_scores, label='Training Scores')
    # plt.plot(range(len(train_scores), len(train_scores) + len(test_scores)), test_scores, label='Testing Scores')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Training and Testing Performance')
    # plt.legend()
    # plt.savefig('Plots/train_test_rewards.png')
    # plt.show()

    # Plot Rewards for Each Agent
    # plt.figure(figsize=(10, 6))
    # plt.figure(figsize=(10, 6))  
    plt.plot(train_rewards[0], label='Agent 1 - Training', linestyle='-', color='black', linewidth=1)
    plt.plot(test_rewards[0], label='Agent 1 - Testing', linestyle=':', color='black',linewidth=1)

    plt.plot(train_rewards[1], label='Agent 2 - Training', color='#495057',linestyle='-',  marker='.',markevery=20,linewidth=1)
    plt.plot(test_rewards[1], label='Agent 2 - Testing', linestyle=':', color='#495057', marker='.',markevery=20,linewidth=1)

    plt.plot(train_rewards[2], label='Agent 3 - Training', color='#6c757d',linestyle='-', marker='x',markevery=20,linewidth=1)
    plt.plot(test_rewards[2], label='Agent 3 - Testing', linestyle=':', color='#6c757d', marker='x',markevery=20,linewidth=1)

    plt.legend(loc=7, fontsize=10, frameon=False) 
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training vs Testing Rewards for Each Agent', fontsize=12)
    plt.grid(False)  
    # plt.tight_layout()  
    plt.savefig('Plots/train_test_comparison_per_agent.png')
    plt.show()

    # plt.plot(train_rewards[0], label='Agent 1 - Training', color='blue')
    # plt.plot(test_rewards[0], label='Agent 1 - Testing', color='blue', linestyle='dashed')

    # plt.plot(train_rewards[1], label='Agent 2 - Training', color='green')
    # plt.plot(test_rewards[1], label='Agent 2 - Testing', color='green', linestyle='dashed')

    # plt.plot(train_rewards[2], label='Agent 3 - Training', color='red')
    # plt.plot(test_rewards[2], label='Agent 3 - Testing', color='red', linestyle='dashed')

    # plt.legend(fontsize=16)
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.title('Training vs Testing Rewards for Each Agent')
    # plt.legend()
    # plt.grid()
    # plt.savefig('Plots/train_test_comparison_per_agent.png')
    # plt.show()
