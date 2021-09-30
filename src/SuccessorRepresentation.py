import matplotlib
import pickle
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import scipy



class SuccessorRepresentation():
                
    def __init__(self, states, occupancy, gamma = 0.9, alpha =0.5):
        ''' An object which computes the successor representation matrix M(s, s') from a set of state-transition observations, and then the value function V based on the SR'''
        
        self.M = np.zeros((len(states), len(states)))
        self.occupancy_seq = occupancy  # a list of sequences of observations
        self.gamma = gamma
        self.alpha = alpha
        

        self.M = np.zeros((len(states), len(states)))


        for seq in self.occupancy_seq:
            for i, s in enumerate(seq[:-1]):
                successor = seq[i+1]
                one_hot = np.zeros(len(states))
                one_hot[s] = 1

                self.M[s, :] += self.alpha * (one_hot + self.gamma*self.M[successor, :] - self.M[s, :])


    def value_function(self, reward_observations):
        ''' Computes the value function as the inner product between the successor matrix and
        the vector with observed rewards
        '''
        
        return self.M @ reward_observations

    def get_SR(self):
        return self.M



