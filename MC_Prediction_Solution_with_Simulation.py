# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:28:54 2017

@author: zhouyonglong
"""

import gym
import matplotlib
import numpy as np
import sys
import pandas as pd

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../")

'''
from lib.envs.blackjack import BlackjackEnv
env = BlackjackEnv()
from lib import plotting
matplotlib.style.use('ggplot')
'''


from Marketing_Env import MarketingEnv
env = MarketingEnv()




def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([x[0] for x in episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Note that in this task the same state never
            # recurs within one episode, so there is no difference 
            # between first-visit and every-visit MC
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,
                     x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            # sum_hand(self.player), self.dealer[0]
            V[state] = returns_sum[state] / returns_count[state]

    return V,returns_sum,returns_count




policy = pd.read_csv('optimal_policy.txt',
                     index_col='state')


def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    '''
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1
    '''
    
    customer_state = observation
    action = policy.loc[customer_state]['策略0']
    return 1 if action=='优惠券' else 0
    

'''
result_10k = mc_prediction(sample_policy, env, num_episodes=10000)
V_10k = result_10k[0]
plotting.plot_value_function(V_10k, title="10,000 Steps")
'''


result_100k = mc_prediction(sample_policy, env, num_episodes=100000)
V_100k = result_100k[0]
sum_v_100k = sum(V_100k.values())


















