import pandas as pd
import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces
import matplotlib.pyplot as plt


# corr = df.corr()
# print(corr.style.background_gradient(cmap='coolwarm'))

#Restrict contrast to 40

class RiseEnv(gym.Env):
  metadata = {'render_modes': ['human','rgb_array'],"render_fps": 4}
  def __init__(self,df,v,render_mode=None):
    #Single agent
    super(RiseEnv,self).__init__()
    self.df = df
    self.v = v

    #########IMPORTANT#########
    # 'shape': (len(self.df),)

    action_info = [
            {'low': self.v['z'][0], 'high': self.v['z'][1]},
            {'low': self.v['f'][0], 'high': self.v['f'][1]},
            {'low': self.v['c'][0], 'high': self.v['c'][1]}]
    self.action_space = [gym.spaces.Box(low=info['low'], high=info['high'], dtype=np.float32) for info in action_info]
    #self.action_space = spaces.Box(low=np.array([self.v['z'][0],self.v['f'][0],self.v['c'][0]]),high=np.array([self.v['z'][1],self.v['f'][1],self.v['c'][1]]))
    observation_info = [
            {'low': self.v['ol'][0], 'high': self.v['ol'][1]},
            {'low': self.v['ec'][0], 'high': self.v['ec'][1]},
            {'low': self.v['at'][0], 'high': self.v['at'][1]},
            {'low': self.v['as'][0], 'high': self.v['as'][1]},
            {'low': self.v['de'][0], 'high': self.v['de'][1]}]
    self.observation_space = [gym.spaces.Box(low=info['low'], high=info['high'], dtype=np.float32) for info in observation_info]
    # self.observation_space = spaces.Box(low=np.array([self.v['ol'][0],self.v['ec'][0],self.v['at'][0],self.v['as'][0],self.v['de'][0]]),high=np.array([self.v['ol'][1],self.v['ec'][1],self.v['at'][1],self.v['as'][1],self.v['de'][1]]), dtype=np.float32)

    self.index = np.random.randint(len(df))
    self.dfOrig = df.iloc[self.index].to_frame().T
    self.dfZ = self.dfOrig.copy().drop(['Focus','Contrast'],axis=1)
    self.dfF = self.dfOrig.copy().drop(['Zoom','Contrast'],axis=1)
    self.dfC = self.dfOrig.copy().drop(['Zoom','Focus'],axis=1)
    self.state = [self.dfZ,self.dfF,self.dfC]
    #Correlations:
    self.gamma_decay = 0.95
    self.gamma = 1.05
    self.low_corr = 0.8
    self.high_corr = 1.2
    self.zoomdeps = {'Orientation_Loss':-0.60,'Edge_Coverage':-0.81,'Average_Thickness':.85,'Average_Separation':0.72}
    self.focusdeps = {'Edge_Coverage':-0.44,'Average_Separation':0.52}

    self.agents = ['Zoom','Focus','Contrast']

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def step(self,actions):

    rewardZ,rewardF,rewardC = 0,0,0
    flagZ,flagF,flagC = [],[],[]

    if self.df['Zoom'].between(self.low_corr*actions[0],self.high_corr*actions[0]).any():
      rewardZ+=10
      # self.dfZ = pd.concat([self.dfZ,])
      self.dfZ = pd.concat([self.dfZ, self.df.loc[self.df['Zoom'].between(self.low_corr*actions[0], self.high_corr*actions[0])].drop(['Focus','Contrast'],axis=1)], ignore_index=True)
    else:
      rewardZ-=1


    if self.df['Focus'].between(self.low_corr*actions[1],self.high_corr*actions[1]).any():
      rewardF+=10
      self.dfF = pd.concat([self.dfF,self.df.loc[self.df['Focus'].between(self.low_corr*actions[1],self.high_corr*actions[1])].drop(['Zoom','Contrast'],axis=1)], ignore_index=True)
    else:
      rewardF-=1


    if self.df['Contrast'].between(self.low_corr*actions[2],self.high_corr*actions[2]).any():
      rewardC+=10
      self.dfC = pd.concat([self.dfC,self.df.loc[self.df['Contrast'].between(self.low_corr*actions[2],self.high_corr*actions[2])].drop(['Zoom','Focus'],axis=1)], ignore_index=True)
    else:
      rewardC-=1

    #Agent-1

    if len(self.dfZ)>1:
      if self.gamma*self.zoomdeps['Orientation_Loss'] <= self.dfZ['Zoom'].corr(self.dfZ['Orientation_Loss']) <= self.gamma_decay*self.zoomdeps['Orientation_Loss']:
        rewardZ+=3
        flagZ.append('a')
      else:
        pass
        # rewardZ-=1
      if self.gamma*self.zoomdeps['Edge_Coverage'] <= self.dfZ['Zoom'].corr(self.dfZ['Edge_Coverage']) <= self.gamma_decay*self.zoomdeps['Edge_Coverage']:
        rewardZ+=3
        flagZ.append('b')
      else:
        pass
        # rewardZ-=1
      if self.gamma_decay*self.zoomdeps['Average_Thickness'] <= self.dfZ['Zoom'].corr(self.dfZ['Average_Thickness']) <= self.gamma*self.zoomdeps['Average_Thickness']:
        rewardZ+=3
        flagZ.append('c')
      else:
        pass
        # rewardZ-=1
      if self.gamma_decay*self.zoomdeps['Average_Separation'] <= self.dfZ['Zoom'].corr(self.dfZ['Average_Separation']) <= self.gamma*self.zoomdeps['Average_Separation']:
        rewardZ+=3
        flagZ.append('d')
      else:
        pass
        # rewardZ-=1

    if flagZ == list(set(['a','b','c','d'])):
      doneZ = True
      flagZ = []
    else:
      doneZ = False

    #Agent-2
    #self.focusdeps = {'Edge_Coverage':-0.44,'Average_Separation':0.52}
    if len(self.dfF)>1:
      if self.gamma*self.focusdeps['Edge_Coverage'] <= self.dfF['Focus'].corr(self.dfF['Edge_Coverage']) <= self.gamma_decay*self.focusdeps['Edge_Coverage']:
        rewardF+=3
        flagF.append('a')
      else:
        pass
        #rewardF-=1

      if self.gamma_decay*self.focusdeps['Average_Separation'] <= self.dfF['Focus'].corr(self.dfF['Average_Separation']) <= self.gamma*self.focusdeps['Average_Separation']:
        rewardF+=3
        flagF.append('b')
      else:
        pass
        #rewardF-=1

      if flagF == list(set(['a','b'])):
        doneF = True
        flagF = []
      else:
        doneF = False

    #Agent-3
    #Contrast has no correlation with any other variables. So, when the other Zoom and Focus agents are done,
    doneC = doneZ and doneF


    next_state = [self.dfZ,self.dfF,self.dfC]
    observations = self._get_obs(self.state)
    info = self._get_info(self.state)
    rewards = [rewardZ,rewardF,rewardC]
    done = [doneZ,doneF,doneC]

    return observations, rewards, done, info, next_state
    

  def _get_info(self,state):
    return {
        "ZoomCorr": state[0].corr() if len(state[0])>1 else None,
        "FocusCorr": state[1].corr() if len(state[0])>1 else None,
        "ContrastCorr": state[2].corr() if len(state[0])>1 else None
    }
  
  def _get_obs(self,state):
    return [np.array(state[i].iloc[:,:5]) for i in range(len(state))]

  def reset(self,seed=None):
    super().reset(seed=seed)
    self.index = np.random.randint(len(self.df))
    self.dfOrig = self.df.iloc[self.index].to_frame().T
    self.state = [self.dfZ,self.dfF,self.dfC]
    return self.state
    

  def render(self):
    pass

  def close(self):
    pass

df = pd.read_excel('Data/FinalData.xlsx')

v= {"ol": [0.5*min(df['Orientation_Loss']),1.5*max(df['Orientation_Loss'])],
        "ec": [0.5*min(df['Edge_Coverage']),1.5*max(df['Edge_Coverage'])],
        "at": [0.5*min(df['Average_Thickness']),1.5*max(df['Average_Thickness'])],
        "as": [0.5*min(df['Average_Separation']),1.5*max(df['Average_Separation'])],
        "de": [0.5*min(df['Distance_Entropy']),1.5*max(df['Distance_Entropy'])],
        "z": [0.5*min(df['Zoom']),1.5*max(df['Zoom'])],
        "f": [0.5*min(df['Focus']),1.5*max(df['Focus'])],
        "c": [0.5*min(df['Contrast']),1.5*max(df['Contrast'])]
        }

env = RiseEnv(df,v)
print(f'env.observation_space: {env.observation_space}')
print(f'env.action_space: {env.action_space}')
print(len(df))
