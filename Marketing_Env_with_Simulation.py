import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np


class MarketingEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
    """
    
    
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(22)
        self._seed()
        
        states_df = pd.read_csv('states.csv',index_col='state')
        self.states = list(states_df.index)
        self.states_dist = list(states_df['percent'])
        
        self.no_groupon_transitions = pd.read_csv('no-groupon-transition.csv',
                                     index_col='state')
        self.groupon_transitions = pd.read_csv('groupon-transition.csv',
                                     index_col='state')
        
        #此处应当读入key为(state,action,reward)的列表，每次从之里面随机选择random.choice
        #所以当确定reward(s,a)时用random.choice不是用 np.random.normal
        self.rewards = pd.read_csv('rewards.csv')

        # Start the first game
        self._reset()        # Number of 
        self.nA = 2
        self.churn_reward = -100
        self.no_actioin_reward = -10

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_reward(self,action):
        reward_mean = self.rewards[(self.rewards['state']==self.current_state) &
                      (self.rewards['action']==action)]['premium_mean']
        reward_std = self.rewards[(self.rewards['state']==self.current_state) &
                              (self.rewards['action']==action)]['premium_std']
        reward = np.random.normal(reward_mean,reward_std,1)[0]
        
        return reward
    
    def _step(self, action):
        if action:  # hit: add a card to players hand and return
            # 如果是model-tree方法，直接从样本中获取next_state
            t = self.groupon_transitions.loc[self.current_state]
            v = list(t.index)
            p = list(t.values)
            #如果是model-tree方法，直接从样本中随机获取state为current_state
            # 动作为action 的 next_state
            next_state = np.random.choice(v,1,p=p)[0]
        
            if next_state=='0':
                
                reward  = self._get_reward(action)
                
                done = True
            elif next_state=='-1':
                reward = self.churn_reward
                done = True
            else:
                reward = self.no_actioin_reward
                done = False
        else:
            t = self.no_groupon_transitions.loc[self.current_state]
            v = list(t.index)
            p = list(t.values)
            next_state = np.random.choice(v,1,p=p)[0]
        
            if next_state=='0':
                
                reward  = self._get_reward(action)
                
                done = True
                
            elif next_state=='-1':
                reward = self.churn_reward
                done = True
            else:
                reward = self.no_actioin_reward
                done = False
                
        self.current_state = next_state
        return self.current_state, reward, done, {}


    def _get_obs(self):
        return self.current_state

    def _reset(self):
        self.current_state = np.random.choice(self.states,1,self.states_dist)[0]
        #self.current_state = np.random.choice(self.states,1)[0]
        return self.current_state




