
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import copy
import scipy
from scipy.optimize import minimize
import pickle
from importlib import reload
import GraphGP

from SuccessorRepresentation import SuccessorRepresentation
from GraphGP import LaplacianGP
import copy
import time






def estimate_successor_model(SR, R, option_indices):
    
    V = SR @ R
    V_i = V[option_indices]
    return V_i[0] - V_i[1]

def estimate_euclidean_model(K, R, training_idx, option_indices):
    
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(euclidean_covariance)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]


def estimate_subjective_euclidean_model(K, R, training_idx, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]

def estimate_GP(K, R, training_idx, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return [options[0], options[1]]
    
def estimate_GP_full(K, R, training_idx):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    return mu

def optimize_gp(X, training_idx, y, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    X_train = X[training_idx]
    K_optimal, l, n = gp.minimize_nll(X, X_train)

    gp.set_covariance(K_optimal)
    mu = gp.mean()
    options = mu[option_indices]
    return [options[0], options[1]]    


def weigh_kernels(k1, k2, training_idx, y):
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

    I = np.eye(len(M))
    jitter = lmbd * np.eye(len(M))
#    M_inv = pseudo_inverse(M)
#    return (M_inv + I)/ - gamma

    T = (np.linalg.inv(M + jitter) - I) / -gamma
    return T

def optimize_diffusion_gp(L, training_idx, y, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    gp.set_laplacian_matrix(L)

    ### implement multiple restarts:
    lengthscale = gp.minimize_nll_diffusion()
    K_optimal = scipy.linalg.expm(-lengthscale*L)

    gp.set_covariance(K_optimal)
    mu = gp.mean()
    options = mu[option_indices]
    return options
#    return options[0] - options[1]    

def compute_entropy(T):
    ''' Computes the entropy of a transition matrix'''
    entropy = 0
    for i, row in enumerate(T):
        T[i] = row/np.sum(row)

    stat_dist = np.linalg.matrix_power(T, 10)
    stationary = stat_dist[0]


    for i, t in enumerate(T):

        t = t[t>0]
        if len(t) == 0:
            continue

        mu = stationary[i]
        e = - np.sum(mu * t * np.log2(t))
        entropy += e

    return entropy

def compute_neumann_entropy(L):
    ''' Computes the neumann entropy of a laplacian matrix'''
    L_ = L/np.trace(L)#np.sum(np.diagonal(L))
    entropy = 0


    eig_vals, eig_vecs = np.linalg.eig(L_)
    eig_vals_prime = eig_vals[eig_vals > 0]
    neumann_entropy = -np.sum(eig_vals_prime*np.log(eig_vals_prime))
    return neumann_entropy

#print(np.log2(0))
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
    
    
def estimate_sr_graph_model(sr_graph, R, training_idx, option_indices, lengthscale):
    
    gp = LaplacianGP()
    gp.train(sr_graph, training_idx, R, alpha=lengthscale)

    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]


def estimate_graph_model(graph, R, training_idx, option_indices, lengthscale):

    gp = LaplacianGP()
    gp.train(graph, training_idx, R, alpha=lengthscale)
    mu = gp.mean()

    options = mu[option_indices]
    return options[0] - options[1]

def RBF(X1, X2, var = 1, l = 1):
        
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return var**2 * np.exp(-0.5 / l**2 * sqdist)


