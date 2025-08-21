import numpy as np
import pandas as pd
from maddpg import MADDPG
import time
from buffer import MultiAgentReplayBuffer
from latest import NewEnv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import csv
from itertools import zip_longest

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
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

# Training MADDPG
def train_maddpg(env, maddpg_agents, n_agents, memory, n_games, print_interval, max_steps=200):
    total_steps = 0
    score_history = []
    reward_history = [[],[],[]]
    evaluate = False

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(n_games):
        obs = env.reset()
        done = [False] * n_agents
        score = 0
        reward1,reward2,reward3 = 0,0,0
        for _ in range(200):
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, _ = env.step(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if all(done):
                break

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            score += sum(reward)
            reward1 += reward[0]
            reward2 += reward[1]
            reward3 += reward[2]
            total_steps += 1

        score_history.append(score)
        reward_history[0].append(reward1)
        reward_history[1].append(reward2)
        reward_history[2].append(reward3)
        if i % print_interval == 0:
            print(f"Episode {i}, Average Score: {np.mean(score_history[-100:])}")
    
    return score_history,reward_history


# Testing MADDPG
def test_maddpg(env, maddpg_agents, n_agents, n_games, max_steps=200):
    test_score_history = []
    reward_history = [[],[],[]]
    for i in range(n_games):
        obs = env.reset()
        done = [False] * n_agents
        score = 0
        reward1,reward2,reward3 = 0,0,0
        for _ in range(200):
            actions = maddpg_agents.choose_action(obs)  
            obs_, reward, done, _ = env.step(actions)
            obs = obs_
            score += sum(reward)
            reward1 += reward[0]
            reward2 += reward[1]
            reward3 += reward[2]

            if all(done):
                break
        
        test_score_history.append(score)
        reward_history[0].append(reward1)
        reward_history[1].append(reward2)
        reward_history[2].append(reward3)
        print(f"Test Episode {i + 1}, Score: {score}")
    
    return test_score_history,reward_history


if __name__ == '__main__':

    scenario = 'rise_simple_spread'
    
    env = NewEnv(df=df,corr_vals=corr_vals)
    n_agents = env.n_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)
    n_actions = [4,4,4]   #Correlations [zoom has 4, focus and contrast have 2, the other 2 are just paddings, that the env dont consider]
    # If the n_actions are of diffent shapes, we need to pad them later as tensor does not accept, so just putting 4,4,4 now and neglecting them in environement

    startTime = time.time()

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.0001, beta=0.0001, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)
    

    print("Training Started...")
    train_env = NewEnv(df=train_df,corr_vals=corr_vals)
    train_scores,train_each_agent = train_maddpg(train_env, maddpg_agents, n_agents, memory, n_games=500, print_interval=50)
    max_score,max_reward1,max_reward2,max_reward3 = max(train_scores),max(train_each_agent[0]),max(train_each_agent[1]),max(train_each_agent[2])
    train_incremental_score_history = [max_score-i for i in train_scores]
    train_incremental_reward1_history = [max_reward1-i for i in train_each_agent[0]]
    train_incremental_reward2_history = [max_reward2-i for i in train_each_agent[1]]
    train_incremental_reward3_history = [max_reward3-i for i in train_each_agent[2]]
    print("Max Training reward: ",max(train_incremental_score_history))
    print("Training Completed!")

    print("Testing Started...")
    test_env = NewEnv(df=test_df,corr_vals=corr_vals)
    test_scores,test_each_agent = test_maddpg(test_env, maddpg_agents, n_agents, n_games=200)
    max_score,max_reward1,max_reward2,max_reward3 = max(test_scores),max(test_each_agent[0]),max(test_each_agent[1]),max(test_each_agent[2])
    test_incremental_score_history = [max_score-i for i in test_scores]
    test_incremental_reward1_history = [max_reward1-i for i in test_each_agent[0]]
    test_incremental_reward2_history = [max_reward2-i for i in test_each_agent[1]]
    test_incremental_reward3_history = [max_reward3-i for i in test_each_agent[2]]
    print("Max Testing reward: ",max(test_incremental_score_history))
    print("Testing Completed!")

    print('It took', time.time()-startTime, 'seconds.')

    #Data to csv
    d1 = [train_incremental_score_history,test_incremental_score_history]
    with open("maddpg.csv","w+") as f:
        w = csv.writer(f)
        for values in zip_longest(*d1):
            w.writerow(values)
    
    d2 = [train_incremental_reward1_history,train_incremental_reward2_history,train_incremental_reward3_history,test_incremental_reward1_history,test_incremental_reward2_history,test_incremental_reward3_history]
    with open("each_agent.csv","w+") as f:
        w = csv.writer(f)
        for values in zip_longest(*d2):
            w.writerow(values)

    # Plot training and testing results
    plt.figure()
    plt.plot(train_incremental_score_history, label='Training Scores', linestyle='solid', color='black')
    plt.plot(test_incremental_score_history, label='Testing Scores', linestyle=':', color='black')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training and Testing Performance', fontsize=12)
    plt.legend(loc='center right', fontsize=10)
    plt.grid(False)
    plt.savefig('Plots/train_test_scores.png')
    plt.show()
    # plt.figure()
    # plt.plot(train_incremental_score_history, label='Training Scores')
    # plt.plot(range(len(train_scores), len(train_scores) + len(test_scores)), test_incremental_score_history, label='Testing Scores')
    # plt.xlabel('Episodes')
    # plt.ylabel('Total Reward')
    # plt.title('Training and Testing Performance')
    # plt.legend()
    # plt.savefig('Plots/train_test_scores.png')
    # plt.show()

    plt.figure()
    plt.plot(train_incremental_reward1_history, label='Agent 1 - Training', linestyle='-', color='black', linewidth=1)
    plt.plot(test_incremental_reward1_history, label='Agent 1 - Testing', linestyle=':', color='black',linewidth=1)

    plt.plot(train_incremental_reward2_history, label='Agent 2 - Training', color='#495057',linestyle='-',  marker='.',markevery=20,linewidth=1)
    plt.plot(test_incremental_reward2_history, label='Agent 2 - Testing', linestyle=':', color='#495057', marker='.',markevery=20,linewidth=1)

    plt.plot(train_incremental_reward3_history, label='Agent 3 - Training', color='#6c757d',linestyle='-', marker='x',markevery=20,linewidth=1)
    plt.plot(test_incremental_reward3_history, label='Agent 3 - Testing', linestyle=':', color='#6c757d', marker='x',markevery=20,linewidth=1)

    plt.legend(loc='center right',fontsize=10, frameon=False) 
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training vs Testing Rewards for Each Agent', fontsize=12)
    plt.grid(False)  
    # plt.tight_layout()  
    plt.savefig('Plots/train_test_comparison_per_agent.png')
    plt.show()

    # plt.figure()
    # plt.plot(train_incremental_reward1_history, label='Agent 1 - Training', color='blue')
    # plt.plot(test_incremental_reward1_history, label='Agent 1 - Testing', color='blue', linestyle='dashed')

    # plt.plot(train_incremental_reward2_history, label='Agent 2 - Training', color='green')
    # plt.plot(test_incremental_reward2_history, label='Agent 2 - Testing', color='green', linestyle='dashed')

    # plt.plot(train_incremental_reward3_history, label='Agent 3 - Training', color='red')
    # plt.plot(test_incremental_reward3_history, label='Agent 3 - Testing', color='red', linestyle='dashed')

    # plt.legend(fontsize=16)
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.title('Training vs Testing Rewards for Each Agent')
    
    # plt.grid()
    # plt.savefig('Plots/train_test_comparison_per_agent.png')
    # plt.show()


     # Plot Training Rewards for Each Agent
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_incremental_reward1_history, label='Agent 1 - Training', color='blue',fontsize=16)
    # plt.plot(train_incremental_reward2_history, label='Agent 2 - Training', color='green')
    # plt.plot(train_incremental_reward3_history, label='Agent 3 - Training', color='red')
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.title('Training Rewards for Each Agent')
    # plt.legend()
    # plt.grid()
    # plt.savefig('Plots/train_rewards_per_agent.png')
    # plt.show()

    # # Plot Testing Rewards for Each Agent
    # # plt.figure(figsize=(10, 6))
    # plt.plot(test_incremental_reward1_history, label='Agent 1 - Testing', color='blue', linestyle='dashed')
    # plt.plot(test_incremental_reward2_history, label='Agent 2 - Testing', color='green', linestyle='dashed')
    # plt.plot(test_incremental_reward3_history, label='Agent 3 - Testing', color='red', linestyle='dashed')
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.title('Testing Rewards for Each Agent')
    # plt.legend()
    # plt.grid()
    # plt.savefig('Plots/test_rewards_per_agent.png')
    # plt.show()

    # Combined Plot: Training and Testing for Each Agent
    # plt.figure(figsize=(10, 6))