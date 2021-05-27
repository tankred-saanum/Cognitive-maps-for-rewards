import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import copy
import scipy
from scipy.optimize import minimize
import pickle
from importlib import reload
from SuccessorRepresentation import SuccessorRepresentation
from GraphGP import LaplacianGP
import copy
import time


'''
Here are a bunch of convenience functions used throughout the multiple analysis
done in this project
'''



def estimate_successor_model(SR, R, option_indices):
    ''' Estimates V with the successor representation'''
    V = SR @ R
    V_i = V[option_indices]
    return V_i[0] - V_i[1]

def estimate_euclidean_model(K, R, training_idx, option_indices):
    ''' Computes difference in estimated value based on Euclidean (at least intended) kernel'''
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(euclidean_covariance)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]



def estimate_GP(K, R, training_idx, option_indices):
    
    ''' gives the estimated values of two options from a GP with a particular kernel'''
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return [options[0], options[1]]
    
def estimate_GP_full(K, R, training_idx):
    '''Gives the full mean function of the GP after conditioning on R'''
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    return mu

def optimize_gp(X, training_idx, y, option_indices):
    ''' optimizes the GPs lengthscale with scipy'''
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    X_train = X[training_idx]
    K_optimal, l, n = gp.minimize_nll(X, X_train)

    gp.set_covariance(K_optimal)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]    


def weigh_kernels(k1, k2, training_idx, y):
    '''Computes weights as the bayesian posterior. This function isnt used in the project
    but could still be an interesting model'''
    gp1 = LaplacianGP()
    gp1.set_training_data(training_idx, y)
    gp1.set_covariance(k1)
    nll1 = gp1.evaluate_nll()
    
    gp2 = LaplacianGP()
    gp2.set_training_data(training_idx, y)
    gp2.set_covariance(k2)
    nll2 = gp2.evaluate_nll()
    
    k1_ml = np.exp(-nll1)
    k2_ml = np.exp(-nll2)

    p1 = k1_ml / (k1_ml + k2_ml)
    p2 = 1-p1
    return p1, p2
    

def estimate_transition_matrix(M, gamma, lmbd = 0.0000001):
    '''This function estimates the transition matrix T from a particular successor matrix M'''

    I = np.eye(len(M))
    jitter = lmbd * np.eye(len(M))

    T = (np.linalg.inv(M + jitter) - I) / -gamma
    return T

def optimize_diffusion_gp(L, training_idx, y, option_indices):
    '''Optimizes the lengthscale of the diffusion kernel with scipy'''
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    gp.set_laplacian_matrix(L)


    lengthscale = gp.minimize_nll_diffusion()
    K_optimal = scipy.linalg.expm(-lengthscale*L)

    gp.set_covariance(K_optimal)
    mu = gp.mean()
    options = mu[option_indices]
    return options



def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def SR_softmax(graph, rewards):
    nodes = list(graph.nodes)
    T = np.zeros((len(nodes),len(nodes) ))
    for i, node in enumerate(nodes):
        p = np.zeros(len(nodes))
        adj = np.array(list(graph.neighbors(node)))
        r_adj = rewards[adj]
        s_max = softmax(r_adj)
        p[ajd] = s_max
        T[i] = p

    T = make_symmetric(T)
    L = np.eye(len(nodes)) - T
    return L
    

def SR_bayesian(graph, prior_T, rewards):
    nodes = list(graph.nodes)
    T = np.zeros((len(nodes),len(nodes) ))
    for i, node in enumerate(nodes):
        p = np.zeros(len(nodes))
        adj = np.array(list(graph.neighbors(node)))
        r_adj = rewards[adj]
        s_max = softmax(r_adj)
        prior = prior_T[i]
        prior /= np.sum(prior)
        p[ajd] = s_max
        p = (p*prior)/np.sum(p*prior)
        T[i] = p

    T = make_symmetric(T)
    L = np.eye(len(nodes)) - T
    return L


def make_symmetric(T):
    '''This function makes the transition matrix symmetric by taking the pairwise maximimum of
    the upper and lower triangular matrix of T'''
    T_upper = np.triu(T)
    T_lower = np.tril(T)

    T_upperT = T_upper.T
    T_lowerT = T_lower.T

    T_upper = np.maximum(T_upper, T_lowerT)
    T_lower = np.maximum(T_upperT, T_lower)
    T = T_upper + T_lower
    return T


def estimate_laplacian(M, gamma, lmbd=0.000001, subj_id=101, plot=False):
    T = estimate_transition_matrix(M, gamma)
    np.fill_diagonal(T, 0)
    T[T<0] = 0
    
    
    ## make matrices symmetric again!
    T_upper = np.triu(T)
    T_lower = np.tril(T)

    T_upperT = T_upper.T
    T_lowerT = T_lower.T

    T_upper = np.maximum(T_upper, T_lowerT)
    T_lower = np.maximum(T_upperT, T_lower)
    T = T_upper + T_lower

    ###
    if plot:
        construct_graph_from_T(T, subj_id)
    ###
    
    L = np.eye(len(T)) - T
 
    return L



def construct_graph_from_T(T, subj_id):
    A = T.copy()
    A[A>0] = 1
    g = nx.from_numpy_matrix(T)
    edges = list(g.edges)
    weights = []
    for e in edges:
        w = T[e[0], e[1]]

        # for smoother colors
        if w < 0.5:
            w+=0.1
        weights.append(w)



    fig, ax = plt.subplots(1, 1)
    plt.title(f"Subject id: {subj_id}")
    nx.draw(g, pos, with_labels=False, edgelist = edges,  width=np.array(weights)*6)
    plt.show()
#    plt.savefig(f"figures/exploration_paths/graphs/{subj_id}.png")
#        nx.draw(g, pos, with_labels=True, edgelist = edges, edge_color = weights,edge_cmap=plt.cm.Greys, width=np.array(weights)*4)
#        plt.colorbar()
#        plt.show()
    
    

def estimate_graph_model(graph, R, training_idx, option_indices, lengthscale):

    gp = LaplacianGP()
    gp.train(graph, training_idx, R, alpha=lengthscale)
    mu = gp.mean()

    options = mu[option_indices]
    return options[0] - options[1]

def RBF(X1, X2, var = 1, l = 1):
        
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return var**2 * np.exp(-0.5 / l**2 * sqdist)


