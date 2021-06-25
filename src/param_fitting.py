import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import scipy
import networkx as nx
import pandas as pd
import pickle
from MonsterPrior import MonsterPrior
from importlib import reload
from MentalMap import MentalMap
from SuccessorRepresentation import SuccessorRepresentation
from GraphGP import LaplacianGP
import copy
import time
from Params import Params

#################################################
#################################################
### This is the file where we generate all the
### model predictors with the various parameter
### settings that we wish to optimize.
### The predictors can be generated seperately.
### If the whole script is run, all datasets
### are generated (this takes a while)
#################################################
#################################################
#################################################
#################################################

### Open pickled files
with open('occupancy_counts.pickle', 'rb') as handle:
    occupancy_dict = pickle.load(handle)


with open('transitions.pickle', 'rb') as handle:
    transition_dict = pickle.load(handle)

with open('subjective_kernel.pickle', 'rb') as handle:
    subjective_kernel_dict = pickle.load(handle)

with open('subjective_grid_search_dict.pickle', 'rb') as handle:
    subj_kernel_grid_dict = pickle.load(handle)


with open('path_integration_kernels.pickle', 'rb') as handle:
    estimated_euclidean_kernels = pickle.load(handle)





### Preamble - set some variables, create dataframes etc, nothing crazy
df = pd.read_csv('choice_data.csv')


subj = np.array(df["subj"])
op1 = np.array(df["option1"])
op2 = np.array(df["option2"])
choices = np.array(df["chosen_object"])
contexts = np.array(df["map"])
decisions = np.array(df["decision"])

# subtract 1 from these vectors so that the monster id becomes indices
op1 -= 1
op2 -= 1
choices -= 1
decisions -= 1

rewards = np.array(df["chosen_value"])

states = np.arange(0, 12)

###############################################
### define the model value estimation functions, these will all be called for each data point
###############################################

def estimate_successor_model(SR, R, option_indices):
    
    V = SR @ R
    V_i = V[option_indices]
    return V_i[0] - V_i[1]

def estimate_euclidean_model(K, R, training_idx, option_indices):
    
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]

def estimate_laplacian(M, gamma, lmbd=0.000001):
    T = estimate_transition_matrix(M, gamma)
    # convert transition matrix to normalized laplacian
    np.fill_diagonal(T, 0)
    T[T<0] = 0  # remove negative entries
    
    ## make matrices symmetric again!
    T_upper = np.triu(T)
    T_lower = np.tril(T)

    T_upperT = T_upper.T
    T_lowerT = T_lower.T

    T_upper = np.maximum(T_upper, T_lowerT)
    T_lower = np.maximum(T_upperT, T_lower)
    T = T_upper + T_lower
    ###
    
    L = np.eye(len(T)) - T
 
    return L
    


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

    

def estimate_subjective_euclidean_model(K, R, training_idx, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]

def estimate_GP_full(K, R, training_idx):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    return mu

def estimate_GP(K, R, training_idx, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]
    


def estimate_transition_matrix(M, gamma, lmbd = 0.0000001):

    I = np.eye(len(M))
    jitter = lmbd * np.eye(len(M))

    T = (np.linalg.inv(M + jitter) - I) / -gamma
    return T

def estimate_sr_graph_model(sr_graph, R, training_idx, option_indices, lengthscale):
    
    gp = LaplacianGP()
    gp.train(sr_graph, training_idx, R, alpha=lengthscale)

    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]


def optimize_diffusion_gp(L, training_idx, y, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    gp.set_laplacian_matrix(L)

    lengthscale = gp.minimize_nll_diffusion()
    K_optimal = scipy.linalg.exp(-lengthscale*L)

    gp.set_covariance(K_optimal)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]    

def optimize_gp(X, training_idx, y, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    X_train = X[training_idx]
    K_optimal, l, n = gp.minimize_nll(X, X_train)

    gp.set_covariance(K_optimal)
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

def make_symmetric(T):
    T_upper = np.triu(T)
    T_lower = np.tril(T)

    T_upperT = T_upper.T
    T_lowerT = T_lower.T

    T_upper = np.maximum(T_upper, T_lowerT)
    T_lower = np.maximum(T_upperT, T_lower)
    T = T_upper + T_lower
    return T

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def SR_softmax(prior_T, rewards):
    nodes = np.arange(len(rewards))
    T = np.zeros((len(nodes),len(nodes)))
    for i, node in enumerate(nodes):
        p = np.zeros(len(nodes))
        adj = prior_T[i]
        adj = np.where(adj > 0)[0]#adj[adj>0]


        r_adj = rewards[adj]
        s_max = softmax(r_adj)
        p[adj] = s_max
        T[i] = p

    T = make_symmetric(T)
    L = np.eye(len(nodes)) - T
    return L, T
    

def SR_bayesian(prior_T, rewards):
    nodes = np.arange(len(rewards))
    T = np.zeros((len(nodes),len(nodes)))
    for i, node in enumerate(nodes):
        p = np.zeros(len(nodes))
        adj = prior_T[i]
        adj = np.where(adj > 0)[0]#adj[adj>0]
        r_adj = rewards[adj]
        s_max = softmax(r_adj)
        prior = prior_T[i]
        prior /= np.sum(prior)
        p[adj] = s_max
        p = (p*prior)/np.sum(p*prior)
        T[i] = p

    T = make_symmetric(T)
    L = np.eye(len(nodes)) - T
    return L, T



    
    
######################
### Create analysis loop
######################

### model possibilities
## Euclidean
## Temporal
## SR
## Compositional
## Optimized Euclidean
## SR softmax


def param_search(params, model, header_is_params, file_name):
    '''
    Function for searching through a set of parameters and saving the estimated value
    difference for each choice (for each parameter setting) in a csv file.

    params: numpy array with the parameter settings
    model: A string specifying model type. Must be:
    - "Euclidean"
    - "Temporal"
    - "SR"
    - "Compositional"
    - "Euclidean-optimized"
    - "SR-softmax"
    header_is_params: Boolean specifying whether the parameter values should be used as header
    file_name: a string specifying the file name
    '''

    
    num_samples = len(params)
    progress_counter = 0

    SR_based = ["Temporal", "SR", "Compositional", "SR-softmax"]
    euclidean_based = ["Euclidean", "Compositional","Euclidean-optimized", "SR-softmax"]

    
    ### this won't be optimized
    sr_diffusion = 1

    ## configure data saving
    if header_is_params:
        
        header = params
        if model == "Euclidean":
            header = np.linspace(0.1, 4, 100)[params]
    else:
        header = np.arange(len(params))
        
    data_frame = np.zeros((len(subj), num_samples))


    ## start loop
    
    for n in range(num_samples):

        print("progress %: ", progress_counter / num_samples)
        progress_counter += 1
        params_i = params[n]

        if model=="Compositional" or model == "SR-softmax":
            ## if model is compositional, unpack the values from 2darray
            learning_rate = params[n, 0]
            lengthscale_index = int(params[n, 1])
            estimated_euclidean = estimated_euclidean_kernels[lengthscale_index]
            


        elif model in SR_based:
            learning_rate = params_i  # in case model is SR based
        elif model in euclidean_based:
            mp = MonsterPrior(np.linspace(0.1, 4, 100)[params_i])
            estimated_euclidean = mp.get_kernel_matrix()
            monster_loc = mp.pos

        
        


        last_subj = -1
        for i, subj_id in enumerate(subj):

            current_context = contexts[i]
            if subj_id != last_subj:

                ### If there's a change in the subject, recompute all the
                ### representations, based on the exploration trials of the subject whose behaviour we seek to model.

                context_dict = {}
                context_dict[1] ={"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))} 
                context_dict[2] = {"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))} 



                ### SR
                if model in SR_based:
                    seq_list = []
                    for run, seq in transition_dict[subj_id].items():
                        seq_ = copy.deepcopy(seq)
                        seq_ -=1
                        seq_list.append(seq_)


                    sr_model = SuccessorRepresentation(states, seq_list, alpha=learning_rate)
                    SR = sr_model.get_SR()

                    SRL = estimate_laplacian(SR, gamma = sr_model.gamma)
                    SR_K = scipy.linalg.expm(-sr_diffusion*SRL)

                if model == "Euclidean":

                    estimated_euclidean_K = estimated_euclidean


                if model == "SR-softmax":
                    SR_dict = {}
                    SR_dict[1] = {}
                    SR_dict[2] = {}

                    T_prior = estimate_transition_matrix(SR, gamma=sr_model.gamma)
                    SR_dict[1]["kernel"] = SR_K
                    SR_dict[2]["kernel"] = SR_K
                    SR_dict[1]["T"] = T_prior
                    SR_dict[2]["T"] = T_prior

                
                if model == "Compositional" or model == "SR-softmax":
                    estimated_euclidean_K = estimated_euclidean[subj_id]                    
                    compositional_kernel = (SR_K + estimated_euclidean_K)/2

                ### add observations for this context
                options = [op1[i], op2[i]]
                choice = choices[i]
                reward = rewards[i]

                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward

                ## set the last subj to the current one
                last_subj = subj_id


            elif len(context_dict[current_context]["rewards"]) == 0:  # check if participant has been able to make any observations in this context yet 
                # if not then let choice be random, and store observations into context dict
                options = [op1[i], op2[i]]
                choice = choices[i]
                reward = rewards[i]

                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward



            else:
                options = [op1[i], op2[i]]
                choice = choices[i]
                reward = rewards[i]

                training_idx = context_dict[current_context]["training_idx"] # the training indices for the gps
                R = copy.copy(context_dict[current_context]["state_rewards"]) # an array with rewards for each state for the SR. We copy so that it doesn't change when we normalize it

                if R.std() != 0:
                    R = (R - R.mean())/R.std()
                else:
                    R = (R - R.mean())

                y = np.array(copy.copy(context_dict[current_context]["rewards"]))  # for use in the gp models. we copy this so we can normalize it and convert it into an array without messing with the original set of reward observations
                if y.std() != 0:
                    y = (y- y.mean())/y.std()

                else:
                    y = (y - y.mean())

                ### sr softmax #####

                if model == "SR-softmax":

                    SR_K = SR_dict[current_context]["kernel"]
                    T_prior = SR_dict[current_context]["T"]

                    K = (SR_K + estimated_euclidean_K)/2
                    rewards_est = estimate_GP_full(K, y, training_idx)
                    diff = rewards_est[options[0]] - rewards_est[options[1]]

                    L_new, T_new = SR_softmax(T_prior, rewards_est)
                    SR_K = scipy.linalg.expm(-1*L_new)
                    SR_dict[current_context]["kernel"] = SR_K
                    SR_dict[current_context]["T"] = T_new
                    T_prior = T_new

                elif model == "Euclidean-optimized":

                    ## optimized gp
                    diff = optimize_gp(monster_loc, training_idx, y, option_indices = options)
                elif model == "SR":
                    ##SR
                    diff = estimate_successor_model(SR, R, option_indices=options)

                elif model == "Euclidean":

                    ### Euclidean
                    diff = estimate_euclidean_model(estimated_euclidean_K, y, training_idx, option_indices=options)

                elif model == "Temporal":
                    ## SR graph
                    diff = estimate_GP(SR_K, y, training_idx, option_indices=options)

                elif model == "Compositional":
                    # compositional model
                    diff = estimate_GP(compositional_kernel, y, training_idx, option_indices=options)


                ### update arrays:
                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward

                ## store data
                data_frame[i, n] = diff
    data_frame = pd.DataFrame(data_frame)
    data_frame.to_csv(f"param_fits/{file_name}", header=header, index=False)


### define hyperparamater search grids and run the search loop
### remember we have the following model strings to input:
##    - "Euclidean"
##    - "Temporal"
##    - "SR"
##    - "Compositional"
##    - "Euclidean-optimized"
##    - "SR-softmax"


## Euclidean

euc_num_samples = 100
euc_params = np.linspace(0, 99, euc_num_samples, dtype=int)
# these are indices for the estimated euclidean kernels with various lengthscale settings,
# estimated individually for each subject. See "compute_euclidean_graphs.py"

param_search(params=euc_params, model="Euclidean", header_is_params=True, file_name="euc_results.csv")

## Temporal
temporal_num_samples = 25  # we use less samples here because the search space is a lot smaller
temporal_params = np.linspace(0.01, 0.7, temporal_num_samples)

param_search(params=temporal_params, model="Temporal", header_is_params=True, file_name="sr_gp_results_v3.csv")

## SR
sr_num_samples = 20
sr_params = np.linspace(0.01, 0.7, sr_num_samples)

param_search(params=sr_params, model="SR", header_is_params=True, file_name="sr_results.csv")

## Compositional
## here the parameter space is 2D, so the number of samples scales quadratically with
## the number of samples in the 1D case. Therefore we restrict ourselves to fewer
## samples in the single dimensions.

comp_num_samples1D = 20 
comp_l = np.linspace(0, 99, comp_num_samples1D, dtype=int)
comp_lr = np.linspace(0.01, 0.7, comp_num_samples1D)
xi, yi = np.meshgrid(comp_lr, comp_l)
comp_params = np.zeros((len(xi.ravel()), 2))
comp_params[:, 0] = xi.ravel()
comp_params[:, 1] = yi.ravel()
comp_num_samples = comp_num_samples1D * comp_num_samples1D

param_search(params=comp_params, model="Compositional", header_is_params=False, file_name="comp_results_v3.csv")

## Optimized Euclidean
## this doesnt use any hyperparameters
op_num_samples = 1
op_params = [1]

param_search(params=op_params, model="Euclidean-optimized", header_is_params=True, file_name="optimized_results.csv")

## SR softmax
## same case as with the compositional model.

softmax_num_samples1D = 20
softmax_l = np.linspace(0, 99, softmax_num_samples1D, dtype=int)
softmax_lr = np.linspace(0.01, 0.7, comp_num_samples1D)
xi, yi = np.meshgrid(softmax_lr, softmax_l)
softmax_params = np.zeros((len(xi.ravel()), 2))
softmax_params[:, 0] = xi.ravel()
softmax_params[:, 1] = yi.ravel()
softmax_num_samples = softmax_num_samples1D * softmax_num_samples1D

param_search(params=softmax_params, model="SR-softmax", header_is_params=False, file_name="softmax_sr.csv")


