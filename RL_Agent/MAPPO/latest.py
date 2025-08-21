import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import pygame

random.seed(42)
dfOrig = pd.read_excel('Data/FinalData.xlsx')
scaler = MinMaxScaler()
df = dfOrig.copy()
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
# print("HII ",corr_vals[0][2])
# print([corr_vals[0][:4]]+[[corr_vals[1][1],corr_vals[1][3]]]+[[corr_vals[2][1],corr_vals[1][2]]])

class NewEnv(gym.Env):
    #  INSTEAD OF ACTIONS AS VALUES OF ZOOM, FOCUS AND CONTRAST, THE ACTIONS ARE CORRELATIONS
    #  WITH THESE CORRELATIONS AND THE DATAFRAME, CONSTRUCT THE ZOOM, FOCUS AND CONTRAST
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,df,corr_vals,render_mode=None, size=5):

        random.seed(42)
        np.random.seed(42)
        self.df = df
        self.corr_vals = corr_vals
        self.agents = ['zoom_agent','focus_agent','contrast_agent']
        self.n_agents = len(self.agents)

        self.state = [x.copy() for x in corr_vals]
        # self.init_vals = [[-100,-100,-100,-100,-100]]*3

        self.action_space =  [spaces.Box(low=-1.0,high=1.0,dtype=np.float32,shape=(4,))]
        # Values of the 5 target parameters which are continuous
        # self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(4,),dtype=np.float32)+2*spaces.Box(low=0.0,high=1.0,shape=(2,),dtype=np.float32)]
        self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(5,),dtype=np.float32)]*3

        self.index = np.random.randint(len(df))

        #Correlations:
        self.gamma_decay = 0.8
        self.gamma = 1.2
        self.gammaRanges = {
            'zol': [1.05,0.95],
            'zec': [1.05,0.95],
            'zat': [0.95,1.05],
            'zas': [0.95,1.05],
            'fec': [0.95,1.05],
            'fas': [1.05,0.95],
            'col': [1.05,0.95],
            'cat': [0.95,1.05]
        }
        self.low_corr = 0.8
        self.high_corr = 1.2
        self.zoomdeps = {'Orientation_Loss':-0.60,'Edge_Coverage':-0.85,'Average_Thickness':.88,'Average_Separation':0.75}
        self.focusdeps = {'Edge_Coverage':0.44,'Average_Separation':-0.52}
        self.contrastdeps = {'Orientation_Loss': -0.3,'Average_Thickness':0.2}
        self.flagZ,self.flagF,self.flagC = [],[],[]
        # self.zoomRange, self.focusRange, self.contrastRange = 0.1, 0.05, 0.02

        #RENDER 
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.size = size
        self.window_size = 512
        self.window = None
        self.clock = None

        #NEW
        self.corr_buffer = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}
    
    def correlations(self,actions,i,factor,param):

        #  THE ACTIONS ARE NOW CORRELATIONS INSTEAD OF ACTUAL VALUES SO DONT USE THIS

        # i: ith action [0,1,2]
        # factor: zoom = 0.2, focus = 0.1, contrast = 0.1
        # param: {"OL": 0, "EC":1, "AT":2, "AS": 3, "DE": 4}
        
        return np.corrcoef(np.array(np.random.uniform((1-factor)*actions[i],(1+factor)*actions[i],len(self.df))).ravel(),self.state[param].ravel())[0][1]

    def step(self,actions):
        
        rewardZ,rewardF,rewardC = 0,0,0
        
        # info = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}

        #Agent- 1
        #OL
       
        if any(self.gammaRanges['zol'][0]*self.zoomdeps['Orientation_Loss'] <= x <= self.gammaRanges['zol'][1]*self.zoomdeps['Orientation_Loss'] for x in actions[0]):
        
            self.corr_buffer['zoom'][0] = min(actions[0].tolist()+[self.corr_buffer['zoom'][0]], key=lambda x:abs(x-self.zoomdeps['Orientation_Loss']))
            rewardZ+=abs(self.zoomdeps['Orientation_Loss']-self.corr_buffer['zoom'][0])
            self.flagZ.append('a')
            self.state[0][0] = self.corr_buffer['zoom'][0]

            #####FROM HERE####
            # self.state[0][0] = min(actions[0].tolist()+[self.init_vals[0][0]], key=lambda x:abs(x-self.zoomdeps['Orientation_Loss']))
            # if abs(self.state[0][0]-self.zoomdeps['Orientation_Loss']) < abs(self.init_vals[0][0]-self.zoomdeps['Orientation_Loss']):
            #     self.flagZ.append('a')
            #     self.init_vals[0][0] = self.state[0][0]
            #     rewardZ += abs(self.state[0][0]-self.zoomdeps['Orientation_Loss'])
                
            #     print("AAA")
        
        #EC
        # if self.gammaRanges['zec'][0]*self.zoomdeps['Edge_Coverage'] <= actions[0][1] <= self.gammaRanges['zec'][1]*self.zoomdeps['Edge_Coverage']:
        if any(self.gammaRanges['zec'][0]*self.zoomdeps['Edge_Coverage'] <= x <= self.gammaRanges['zec'][1]*self.zoomdeps['Edge_Coverage'] for x in actions[0]):

            self.corr_buffer['zoom'][1] = min(actions[0].tolist()+[self.corr_buffer['zoom'][1]], key=lambda x:abs(x-self.zoomdeps['Edge_Coverage']))
            rewardZ+=abs(self.zoomdeps['Edge_Coverage']-self.corr_buffer['zoom'][1])
            self.flagZ.append('b')
            self.state[0][1] = self.corr_buffer['zoom'][1]

            #####  FROM HERE  ####
            # self.state[0][1] = min(actions[0].tolist()+[self.init_vals[0][1]], key=lambda x:abs(x-self.zoomdeps['Edge_Coverage']))
            # if abs(self.state[0][1]-self.zoomdeps['Edge_Coverage']) < abs(self.init_vals[0][1]-self.zoomdeps['Edge_Coverage']):
            #     self.init_vals[0][1] = self.state[0][1]
            #     rewardZ+= abs(self.state[0][1]-self.zoomdeps['Edge_Coverage'])
                
            #     self.flagZ.append('b')
            #     print("BBB")
        
        #AT
        # if self.gammaRanges['zat'][0]*self.zoomdeps['Average_Thickness'] <= actions[0][2] <= self.gammaRanges['zat'][1]*self.zoomdeps['Average_Thickness']:
        if any(self.gammaRanges['zat'][0]*self.zoomdeps['Average_Thickness'] <= x <= self.gammaRanges['zat'][1]*self.zoomdeps['Average_Thickness'] for x in actions[0]):
        
            
            self.corr_buffer['zoom'][2] = min(actions[0].tolist()+[self.corr_buffer['zoom'][2]], key=lambda x:abs(x-self.zoomdeps['Average_Thickness']))
            rewardZ+=abs(self.zoomdeps['Average_Thickness']-self.corr_buffer['zoom'][2])
            self.flagZ.append('c')
            self.state[0][2] = self.corr_buffer['zoom'][2]

            #####  FROM HERE  ####
            # self.state[0][2] = min(actions[0].tolist()+[self.init_vals[0][2]], key=lambda x:abs(x-self.zoomdeps['Average_Thickness']))
            # if abs(self.state[0][2]-self.zoomdeps['Average_Thickness']) < abs(self.init_vals[0][2]-self.zoomdeps['Average_Thickness']):
            #     self.init_vals[0][2] = self.state[0][2]
            #     rewardZ+=abs(self.state[0][2]-self.zoomdeps['Average_Thickness'])
                
            #     self.flagZ.append('c')
            #     print("CCC")
        
        #AS
        # if self.gammaRanges['zas'][0]*self.zoomdeps['Average_Separation'] <= actions[0][3] <= self.gammaRanges['zas'][1]*self.zoomdeps['Average_Separation']:
        if any(self.gammaRanges['zas'][0]*self.zoomdeps['Average_Separation'] <= x <= self.gammaRanges['zas'][1]*self.zoomdeps['Average_Separation'] for x in actions[0]):
           
            self.corr_buffer['zoom'][3] = min(actions[0].tolist()+[self.corr_buffer['zoom'][3]], key=lambda x:abs(x-self.zoomdeps['Average_Separation']))
            rewardZ+=abs(self.zoomdeps['Average_Separation']-self.corr_buffer['zoom'][3])
            self.flagZ.append('d')
            self.state[0][3] =  self.corr_buffer['zoom'][3]

            #####  FROM HERE  ####
            # self.state[0][3] = min(actions[0].tolist()+[self.init_vals[0][3]], key=lambda x:abs(x-self.zoomdeps['Average_Separation']))
            # if abs(self.state[0][3]-self.zoomdeps['Average_Separation']) < abs(self.init_vals[0][3]-self.zoomdeps['Average_Separation']):
            #     self.init_vals[0][3] = self.state[0][3]
            #     rewardZ+=abs(self.state[0][3]-self.zoomdeps['Average_Separation'])
                
            #     self.flagZ.append('d')
            #     print("DDD")
        

        # if sorted(list(set(self.flagZ))) == ['a','b','c','d']:
        if all(valZ in self.flagZ for valZ in ['a','b','c','d']):
            
            doneZ = True
            # rewardZ+=1
            self.flagZ = []
            # self.init_vals[0] = [-100,-100,-100,-100]
            
        else:
            doneZ = False
        

        #Agent-2
        #EC
        # if self.gammaRanges['fec'][0]*self.focusdeps['Edge_Coverage'] <= actions[1][0] <= self.gammaRanges['fec'][1]*self.focusdeps['Edge_Coverage']:
        if any(self.gammaRanges['fec'][0]*self.focusdeps['Edge_Coverage'] <= x <= self.gammaRanges['fec'][1]*self.focusdeps['Edge_Coverage'] for x in actions[1]):
            
            self.corr_buffer['focus'][0] = min(actions[1].tolist()[:2]+[self.corr_buffer['focus'][0]], key=lambda x:abs(x-self.focusdeps['Edge_Coverage']))
            rewardF+=abs(self.focusdeps['Edge_Coverage']-self.corr_buffer['focus'][0])*2
            self.flagF.append('e')
            self.state[1][1] = self.corr_buffer['focus'][0]

            #####  FROM HERE  ####

            # self.state[1][1] = min(actions[1].tolist()[:2]+[self.init_vals[1][1]], key=lambda x:abs(x-self.focusdeps['Edge_Coverage']))
            # if abs(self.state[1][1]-self.focusdeps['Edge_Coverage']) < abs(self.init_vals[1][1]-self.focusdeps['Edge_Coverage']):
            #     self.init_vals[1][1] = self.state[1][1]
            #     self.flagF.append('e')
            #     # rewardF+= abs(self.state[1][1]-self.focusdeps['Edge_Coverage'])
            #     # rewardF+= 1.5
            #     print(5)
        
        #AS
        # if self.gammaRanges['fas'][0]*self.focusdeps['Average_Separation'] <= actions[1][1] <= self.gammaRanges['fas'][1]*self.focusdeps['Average_Separation']:
        if any(self.gammaRanges['fas'][0]*self.focusdeps['Average_Separation'] <= x <= self.gammaRanges['fas'][1]*self.focusdeps['Average_Separation'] for x in actions[1]):
            
            self.corr_buffer['focus'][1] = min(actions[1].tolist()[:2]+[self.corr_buffer['focus'][1]], key=lambda x:abs(x-self.focusdeps['Average_Separation']))
            rewardF+=abs(self.focusdeps['Average_Separation']-self.corr_buffer['focus'][1])*2
            self.flagF.append('f')
            self.state[1][3] = self.corr_buffer['focus'][1]

            #####  FROM HERE  ####

            # self.state[1][3] = min(actions[1].tolist()[:2]+[self.init_vals[1][3]], key=lambda x:abs(x-self.focusdeps['Average_Separation']))
            # if abs(self.state[1][3]-self.focusdeps['Average_Separation']) < abs(self.init_vals[1][3]-self.focusdeps['Average_Separation']):
            #     self.init_vals[1][3] = self.state[1][3]
            #     self.flagF.append('f')
            #     # rewardF+=abs(self.state[1][3])*1.5
            #     # rewardF+= 1.5
            #     print(6)
        
        if all(valZ in self.flagZ for valZ in ['e','f']):
            doneF = True
            self.flagF = []
            # self.init_vals[1] = [-100,-100,-100,-100]
            # rewardF+= 1.5
        else:
            doneF = False
        
        #Agent-3
        #OL
        # if self.gammaRanges['col'][0]*self.contrastdeps['Orientation_Loss'] <= actions[2][0] <= self.gammaRanges['col'][1]*self.contrastdeps['Orientation_Loss']:
        if any(self.gammaRanges['col'][0]*self.contrastdeps['Orientation_Loss'] <= x <= self.gammaRanges['col'][1]*self.contrastdeps['Orientation_Loss'] for x in actions[2]):
            # rewardC+= abs(actions[2][0])*2
            # rewardC+= abs(self.contrastdeps['Orientation_Loss'])*3
            # rewardC+=.6
            # info['contrast'][0] = min(actions[2].tolist()[:2]+[info['contrast'][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            # self.state[2][0] = info['contrast'][0]

            self.corr_buffer['contrast'][0] = min(actions[2].tolist()[:2]+[self.corr_buffer['contrast'][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            rewardC+=abs(self.contrastdeps['Orientation_Loss']-self.corr_buffer['contrast'][0])*2
            self.flagC.append('g')
            self.state[2][0] = self.corr_buffer['contrast'][0]

            #####  FROM HERE  ####

            # self.state[2][0] = min(actions[2].tolist()[:2]+[self.init_vals[2][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            # if abs(self.state[2][0]-self.contrastdeps['Orientation_Loss']) < abs(self.init_vals[2][0]-self.contrastdeps['Orientation_Loss']):
            #     self.init_vals[2][0] = self.state[2][0]
            #     self.flagC.append('g')
            #     # rewardC+=abs(self.state[2][0])*2
            #     # rewardC+= 2
            #     print(7)


        #AT
        # if self.gammaRanges['cat'][0]*self.contrastdeps['Average_Thickness'] <= actions[2][1] <= self.gammaRanges['cat'][1]*self.contrastdeps['Average_Thickness']:
        if any(self.gammaRanges['cat'][0]*self.contrastdeps['Average_Thickness'] <= x <= self.gammaRanges['cat'][1]*self.contrastdeps['Average_Thickness'] for x in actions[2]):
            # rewardC+= abs(actions[2][1])*2
            # rewardC+= abs(self.contrastdeps['Average_Thickness'])*3
            # rewardC+=.6
            # info['contrast'][1] = min(actions[2].tolist()[:2]+[info['contrast'][1]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            # self.state[2][1] = info['contrast'][1]
            

            self.corr_buffer['contrast'][1] = min(actions[2].tolist()[:2]+[self.corr_buffer['contrast'][1]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            rewardC+=abs(self.contrastdeps['Average_Thickness']-self.corr_buffer['contrast'][1])*2
            self.flagC.append('h')
            self.state[2][2] = self.corr_buffer['contrast'][1]

            #####  FROM HERE  ####

            # self.state[2][2] = min(actions[2].tolist()[:2]+[self.init_vals[2][2]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            # if abs(self.state[2][2]-self.contrastdeps['Average_Thickness']) < abs(self.init_vals[2][2]-self.contrastdeps['Average_Thickness']):
            #     self.init_vals[2][2] = self.state[2][2]
            #     # rewardC+=abs(self.state[2][2])*2
            #     # rewardC+= 2
            #     self.flagC.append('h') 
            #     print(8)
        
        if all(valZ in self.flagZ for valZ in ['g','h']):
            doneC= True
            # rewardC+= 2
            self.flagC= []
            # self.init_vals[2] =[-100]*4
        else:
            doneC= False
        
        #correlations function calculates the correlation between two arrays,so need to update the state based on agent actions 
        #within the step function. Using actions to calculate the correlations. 
            

        observations = self.state
        # print(observations)
        
        return observations, [rewardZ,rewardF,rewardC], [doneZ,doneF,doneC], {}
        # print("Hiiiiiii", observations)
        # return observations, [rewardZ,rewardF,rewardC], [False,False,False], {}

        


    def reset(self):
        state = self.corr_vals
        # self.init_vals = [[-100,-100,-100,-100,-100]]*3
        # self.corr_buffer = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}
        return state


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        red = (255, 0, 0)
        blue = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (255, 255, 0)

        param_positions = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
        for pos in param_positions:
            pygame.draw.circle(self.window, red, pos, 10)

    
        agent_colors = [blue, green, yellow]
        agent_positions = [(50 + int(self.state[0][i] * 450), 50 + int(self.state[0][i] * 450)) for i in range(4)]
        agent_positions += [(50 + int(self.state[1][i] * 450), 50 + int(self.state[1][i] * 450)) for i in range(2)]
        agent_positions += [(50 + int(self.state[2][i] * 450), 50 + int(self.state[2][i] * 450)) for i in range(2)]

        # Draw agents
        for i, pos in enumerate(agent_positions):
            color = agent_colors[i // 4] 
            pygame.draw.circle(self.window, color, pos, 10)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_frame(self):
        self.render()

        