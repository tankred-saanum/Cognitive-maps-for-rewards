import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.stats
import networkx as nx
import pandas as pd
import pickle
from MonsterPrior import MonsterPrior
from importlib import reload
from pathlib import Path

from MentalMap import MentalMap
from SuccessorRepresentation import SuccessorRepresentation
from GraphGP import LaplacianGP
import copy
import time
from Params import Params


from utils import *

### In this file I simulate behaviour from models we're interested in
### The synthetic behaviour is analysed in an R-file.

### here define main simulation function:

def random_choice(options, reward_matrix, current_context):
    
    choice_idx = np.random.randint(0, 1)
    choice = options[choice_idx]
    unchosen = options[1-choice_idx]
    reward = reward_matrix[choice, current_context -1]
    other_reward = reward_matrix[unchosen, current_context -1]
    regret = np.max([reward, other_reward]) - reward

    return choice, choice_idx, reward, regret

def directed_choice(options, reward_matrix, current_context, values):
    
    choice_idx = np.argmax(values)
    choice = options[choice_idx]
    unchosen = options[1-choice_idx]
    reward = reward_matrix[choice, current_context -1]
    other_reward = reward_matrix[unchosen, current_context -1]
    regret = np.max([reward, other_reward]) - reward

    return choice, choice_idx, reward, regret




def simulate_choice_data(df, reward_matrix, params, PI_dict, transition_dict, method ,optimize_l = False):
    ''' specify which model you want to use. the function returns an array of actions,
    rewards, regret, rpes etc.'''

    last_subj = -1  # make this an id so that the first participant isn't identical to this one
    subj_counter = -1

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
    states = np.arange(0, 12)
    ## set hyperparameters
    learning_rate = params["learning_rate"]
    lengthscale = params["lengthscale"]
    sr_diffusion = 1
    
    choices = np.zeros(len(subj))
    choice_indices = np.zeros(len(subj))
    gains = np.zeros(len(subj))
    regret = np.zeros(len(subj))
    RPE = np.zeros(len(subj))
    
    for i, subj_id in enumerate(subj):
        current_context = contexts[i]
        if subj_id != last_subj:

            subj_counter += 1
            trial_counter = 0
            loc = PI_dict[subj_id]

            context_dict = {}
            context_dict[1] ={"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))} #copy.deepcopy(dict_template)
            context_dict[2] = {"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))} #copy.deepcopy(dict_template)

            ### SR

            if method == "SR-GP" or method == "Compositional":
                seq_list = []
                for run, seq in transition_dict[subj_id].items():
                    seq_ = copy.deepcopy(seq)
                    seq_ -=1
                    seq_list.append(seq_)

                    sr_model = SuccessorRepresentation(states, seq_list, alpha=learning_rate)
                    SR = sr_model.get_SR()


                    SRL = estimate_laplacian(SR, gamma = sr_model.gamma, subj_id = subj_id, plot=False)
                    SR_kernel = scipy.linalg.expm(-sr_diffusion*SRL)

        
            if method == "Euclidean" or method == "Compositional":            
                estimated_euclidean_kernel = RBF(loc, loc, l=lengthscale)

            if method == "Compositional":
                comp_kernel = (estimated_euclidean_kernel + SR_kernel)/2


            ### add observations for this context
            options = [op1[i], op2[i]]
            choice, c_idx, reward, regret_i = random_choice(options, reward_matrix, current_context)
            
            choices[i] = choice
            choice_indices[i] = c_idx            
            gains[i] = reward
            regret[i] = regret_i
            RPE[i] =  - reward            

            context_dict[current_context]["training_idx"].append(choice)
            context_dict[current_context]["rewards"].append(reward)

            
            ## set the last subj_id to the current one
            last_subj = subj_id
            trial_counter += 1


        elif len(context_dict[current_context]["rewards"]) == 0:  # check if participant has been able to make any observations in this context yet 
            # if not then let choice be random, and store observations into context dict

            options = [op1[i], op2[i]]
            choice, c_idx, reward, regret_i = random_choice(options, reward_matrix, current_context)

            choices[i] = choice
            choice_indices[i] = c_idx            
            gains[i] = reward
            regret[i] = regret_i
            RPE[i] =  - reward            

            
            context_dict[current_context]["training_idx"].append(choice)
            context_dict[current_context]["rewards"].append(reward)
                        
            trial_counter += 1


        else:
            options = [op1[i], op2[i]]
            training_idx = context_dict[current_context]["training_idx"] # the training indices for the gps
            y = np.array(copy.copy(context_dict[current_context]["rewards"]))  # for use in the gp models. we copy this so we can normalize it and convert it into an array without messing with the original set of reward observations

            y_prime = np.append(y, reward)

            if y.std() != 0:
                y = (y- y.mean())/y.std()
                y_prime = (y_prime - y_prime.mean())/y_prime.std()

            else:
                y = (y - y.mean())
                y_prime = (y_prime - y_prime.mean())

            reward_normalized = y_prime[-1]
            ### Euclidean prediction error
            if method == "SR-GP":
                if optimize_l:
                    preds = optimize_diffusion_gp(SRL, training_idx, y, option_indices = options)
                else:
                    preds = estimate_GP(SR_kernel, y, training_idx, option_indices=options)
            elif method == "Euclidean":
                                    
                preds = estimate_GP(estimated_euclidean_kernel, y, training_idx, option_indices=options)

            elif method == "Mean-tracker":
                BMT_kernel = np.eye(12)
                preds = estimate_GP(BMT_kernel, y, training_idx, option_indices=options)
            else:                
                preds = estimate_GP(comp_kernel, y, training_idx, option_indices=options)
                
            ###
            choice, c_idx, reward, regret_i = directed_choice(options, reward_matrix, current_context, preds)
            
            choices[i] = choice
            choice_indices[i] = c_idx
            gains[i] = reward
            regret[i] = regret_i
            RPE[i] = preds[c_idx] - reward_normalized

            ### update arrays:
            context_dict[current_context]["training_idx"].append(choice)
            context_dict[current_context]["rewards"].append(reward)
            context_dict[current_context]["state_rewards"][choice] = reward

            trial_counter += 1

    results = np.zeros((len(subj), 5))
    results[:, 0] = choices
    results[:, 1] = choice_indices
    results[:, 2] = gains
    results[:, 3] = regret
    results[:, 4] = RPE

    results_df = pd.DataFrame(results)
    
    return results_df
    




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

with open('path_integration_monster_locations.pickle', 'rb') as handle:
    PI_dict = pickle.load(handle)


### Unpack choice data and create reward matrices

df = pd.read_csv('choice_data.csv')
r_df = pd.read_csv("moster_rewards.csv")
reward_matrix = np.zeros((12, 2))
reward_matrix[:, 0] = r_df["ctx1"]
reward_matrix[:, 1] = r_df["ctx2"]


params = {"lengthscale": 2.05, "learning_rate": 0.4}
header = ["choices", "choice_indices", "rewards", "regret", "RPE"]

################# Run simulations
#################################

########EUCLIDEAN:
##################

euc_sim = simulate_choice_data(df, reward_matrix, params, PI_dict, transition_dict, method="Euclidean", optimize_l = False)
euc_sim.to_csv("model_simulations/simulations/euclidean_sim.csv", index=False, header=header)

########SR-GP:
##############

sr_gp_sim = simulate_choice_data(df, reward_matrix, params, PI_dict, transition_dict, method="SR-GP", optimize_l = False)
sr_gp_sim.to_csv("model_simulations/simulations/sr_gp_sim.csv", index=False, header=header)

########SR-GP-OPTIMIZED:
########################

sr_gp_sim_optimized = simulate_choice_data(df, reward_matrix, params, PI_dict, transition_dict, method="SR-GP", optimize_l = True)
sr_gp_sim_optimized.to_csv("model_simulations/simulations/sr_gp_sim_optimized.csv", index=False, header=header)

########COMPOSITIONAL:
######################

compositional_sim = simulate_choice_data(df, reward_matrix, params, PI_dict, transition_dict, method="Compositional", optimize_l = False)
compositional_sim.to_csv("model_simulations/simulations/compositional_sim.csv", index=False, header=header)

########MEAN-TRACKER:
#####################

mean_tracker_sim = simulate_choice_data(df, reward_matrix, params, PI_dict, transition_dict, method="Mean-tracker", optimize_l = False)
mean_tracker_sim.to_csv("model_simulations/mean_tracker_sim.csv", index=False, header = header)


