import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
from collections import Counter

class MentalMap():
    def __init__(self, state_sequence=None):

        self.state_sequence = state_sequence

    def generate_graph_from_matrix(self, matrix, t = 0.):
        
        ''' 
        Generates a graph based on a n x n covariance/transition matrix. The weight of the edge is 
        a function of how the entry in the matrix. If threshold t > 0, then entries must be higher than
        t in order to produce an edge in the graph. Else, all nodes are connected (but weighted by 
        matrix entry)
        '''

        nodes = np.arange(len(matrix))
        G = nx.Graph()
        G.add_nodes_from(nodes)

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if matrix[i, j] > t and i != j:
                    e = (node_i, node_j)#, {'weight': matrix[i, j]})
                    G.add_edge(*e)
                    G[node_i][node_j]['weight'] = matrix[i, j]

        self.graph = G
        return self.graph


    def generate_graph(self, weighted = False):
        ''' generates a graph based on observed transitions from an
        exploration run, but does not take transition counts into the 
        equation'''
        
        transition_tuples = []
 #       transition_sets = []
        episode_length = len(self.state_sequence) - 1 # the last index of state_sequence
 #       counts = []

        for i, state in enumerate(self.state_sequence):
            if i + 1 <= episode_length:  #check if successor is lower than last idx
                successor = self.state_sequence[i+1]
                transition = (state, successor)
                transition_tuples.append(transition)


        nodes = np.arange(12)  # then number of monsters i.e. states
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        if weighted:
            for i, transition in enumerate(transition_tuples):
                # check if edge exists
                if G.has_edge(transition[0], transition[1]):
                    G[transition[0]][transition[1]]["weight"] += 1
                else:
                    G.add_edge(transition[0], transition[1], weight = 1)
        else:
            G.add_edges_from(transition_tuples)

        self.graph = G
        return self.graph


    def get_graph(self):
        return self.graph

    def get_graph_laplacian(self):
        return np.array(nx.laplacian_matrix(self.graph).todense())

