from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import networkx as nx
import pandas as pd
import pickle
from MonsterPrior import MonsterPrior
from importlib import reload
from SuccessorRepresentation import SuccessorRepresentation
from GraphGP import LaplacianGP
import copy
import time




def count_transitions(seq):
    num_monsters = 12
    T = np.zeros((num_monsters, num_monsters))
    if len(seq) == 0:
        return T
    last_monster = seq[0]
    for monster in seq[1:]:
        T[last_monster, monster] += 1
        last_monster = monster

    return T

def make_symmetric(T):
    T_upper = np.triu(T)
    T_lower = np.tril(T)

    T_upperT = T_upper.T
    T_lowerT = T_lower.T

    T_upper = np.maximum(T_upper, T_lowerT)
    T_lower = np.maximum(T_upperT, T_lower)
    T = T_upper + T_lower
    return T



def run_transition_counts_model(num_samples, file_name="transition_counts_model.csv"):


    progress_counter = 0
    num_monsters = 12
    data_frame = np.zeros((len(subj), num_samples))


    ## start loop
    lengthscales = np.linspace(0.1, 4, num_samples)
    header = lengthscales
    for n in range(num_samples):
        lengthscale = lengthscales[n]
        #lengthscale = 1
        last_subj = -1
        for i, subj_id in enumerate(subj):

            current_context = contexts[i]
            if subj_id != last_subj:

                ### If there's a change in the subject, recompute all the
                ### representations, based on the exploration trials of the subject whose behaviour we seek to model.

                context_dict = {}
                context_dict[1] ={"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))}
                context_dict[2] = {"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))}


                T = np.zeros((num_monsters, num_monsters))
                for k, (run, seq) in enumerate(transition_dict[subj_id].items()):

                        seq_ = copy.deepcopy(seq)
                        seq_ -=1

                        T_ = count_transitions(seq_)
                        T += T_
                np.fill_diagonal(T, 0)
                for i, row in enumerate(T):
                    if row.sum() != 0:
                        T[i] = row/row.sum()

                T = make_symmetric(T)

                L = np.eye(num_monsters) - T
                kernel = scipy.linalg.expm(-lengthscale*L)
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

                diff = estimate_euclidean_model(kernel, y, training_idx, option_indices=options)



                ### update arrays:
                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward

                ## store data
                data_frame[i, n] = diff


    data_frame = pd.DataFrame(data_frame)
    data_frame.to_csv(f"param_fits/{file_name}", header=header, index=False)

def estimate_euclidean_model(K, R, training_idx, option_indices):

    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean()
    options = mu[option_indices]
    return options[0] - options[1]


with open('transitions.pickle', 'rb') as handle:
    transition_dict = pickle.load(handle)


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

run_transition_counts_model(num_samples=1, file_name="transition_counts_model.csv")
