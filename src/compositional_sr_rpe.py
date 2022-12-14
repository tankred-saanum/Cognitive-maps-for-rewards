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

    if (k1_ml + k2_ml) != 0:
        p1 = k1_ml / (k1_ml + k2_ml)
        p2 = 1-p1
    else:
        p1 = 0.5
        p2 = 0.5
    return p1, p2


def get_ml(k1, k2, training_idx, y):
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


    return k1_ml, k2_ml

def estimate_GP(K, R, training_idx, option_indices):
    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean(sigma=0.001)
    options = mu[option_indices]
    return [options[0], options[1]]

def estimate_transition_matrix(M, gamma, lmbd = 0.0000001):

    I = np.eye(len(M))
    jitter = lmbd * np.eye(len(M))

    T = (np.linalg.inv(M + jitter) - I) / -gamma
    return T


def estimate_euclidean_model(K, R, training_idx, option_indices):

    gp = LaplacianGP()
    gp.set_training_data(training_idx, R)
    gp.set_covariance(K)
    mu = gp.mean(sigma=0.001)
    options = mu[option_indices]
    return options[0] - options[1]

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


def RBF(X1, X2, var = 1, l = 1):

    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return var**2 * np.exp(-0.5 / l**2 * sqdist)
    ## start loop


def run_compositional_sr_model(num_samples, file_name="comp_sr_model.csv", path_integration = True, optimize_lr = False, optimize_post_weight=False):


    progress_counter = 0
    num_monsters = 12
    data_frame = np.zeros((len(subj), num_samples*num_samples))
    weights_rpe = np.zeros_like(data_frame)
    RPEs_euc = np.zeros_like(data_frame)
    RPEs_temp = np.zeros_like(data_frame)
    weights_ml = np.zeros_like(data_frame)
    ml_euc = np.zeros_like(data_frame)
    ml_temp = np.zeros_like(data_frame)
    RPEs_comp = np.zeros_like(data_frame)
    states = np.arange(0, 12)

    ## start loop
    lengthscales = np.linspace(0.1, 4, num_samples)
    learning_rates = np.linspace(0.001, 0.5, num_samples)
    post_weights = np.linspace(0.01, 0.99, num_samples)
    if optimize_post_weight:
        l_x, l_y = np.meshgrid(lengthscales, post_weights)
        hparams = np.array([l_x.ravel(), l_y.ravel()]).T
    else:
        if not optimize_lr:
            l_x, l_y = np.meshgrid(lengthscales, lengthscales)
            hparams = np.array([l_x.ravel(), l_y.ravel()]).T
        else:
            l_x, l_y = np.meshgrid(lengthscales, learning_rates)
            hparams = np.array([l_x.ravel(), l_y.ravel()]).T
    header = np.arange(len(hparams))
    for n in range(len(hparams)):

        hparam_euc, hparam_temp = hparams[n, 0], hparams[n, 1]
        if optimize_post_weight:
            w_post = hparam_temp
        #lengthscale = 1
        last_subj = -1
        print("Progress: ", progress_counter / len(hparams), end = "\r")
        progress_counter += 1
        for i, subj_id in enumerate(subj):

            current_context = contexts[i]
            if subj_id != last_subj:
                w_euc = 0.5
                p_euc = 0.5

                ### If there's a change in the subject, recompute all the
                ### representations, based on the exploration trials of the subject whose behaviour we seek to model.

                context_dict = {}
                context_dict[1] ={"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))}
                context_dict[2] = {"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))}


                seq_list = []
                for run, seq in transition_dict[subj_id].items():
                    seq_ = copy.deepcopy(seq)
                    seq_ -=1
                    seq_list.append(seq_)



                if not optimize_lr:
                    sr_model = SuccessorRepresentation(states, seq_list, alpha=0.001)
                    SR = sr_model.get_SR()

                    SRL = estimate_laplacian(SR, gamma = sr_model.gamma)
                    kernel_temp = scipy.linalg.expm(-hparam_temp*SRL)
                else:
                    sr_model = SuccessorRepresentation(states, seq_list, alpha=hparam_temp)
                    SR = sr_model.get_SR()

                    SRL = estimate_laplacian(SR, gamma = sr_model.gamma)
                    kernel_temp = scipy.linalg.expm(-1*SRL)

                if path_integration:
                    loc = PI_dict[subj_id]
                else:
                    mp = MonsterPrior()
                    loc = mp.pos
                kernel_space = RBF(loc, loc, l=hparam_euc)
                kernel = ((kernel_space*w_euc) + (kernel_temp*(1-w_euc)))# / 2
                kernel_comp_unweighted = (kernel_space + kernel_temp) / 2

                ### add observations for this context
                options = [op1[i], op2[i]]
                choice = choices[i]
                reward = rewards[i]

                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward

                ## set the last subj to the current one
                last_subj = subj_id

                weights_rpe[i, n] = w_euc
                RPEs_euc[i, n] = reward
                RPEs_temp[i, n] = reward
                RPEs_comp[i, n] = reward
                weights_ml[i, n] = p_euc


            elif len(context_dict[current_context]["rewards"]) == 0:  # check if participant has been able to make any observations in this context yet
                # if not then let choice be random, and store observations into context dict
                options = [op1[i], op2[i]]
                choice = choices[i]
                reward = rewards[i]

                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward

                weights_rpe[i, n] = w_euc
                RPEs_euc[i, n] = reward
                RPEs_temp[i, n] = reward
                RPEs_comp[i, n] = reward
                weights_ml[i, n] = p_euc


            else:
                options = [op1[i], op2[i]]
                choice = choices[i]
                reward = rewards[i]
                decision = decisions[i]

                training_idx = context_dict[current_context]["training_idx"] # the training indices for the gps
                R = copy.copy(context_dict[current_context]["state_rewards"]) # an array with rewards for each state for the SR. We copy so that it doesn't change when we normalize it

                if R.std() != 0:
                    R = (R - R.mean())/R.std()
                else:
                    R = (R - R.mean())

                y = np.array(copy.copy(context_dict[current_context]["rewards"]))  # for use in the gp models. we copy this so we can normalize it and convert it into an array without messing with the original set of reward observations
                y_prime = np.append(y, reward)
                if y.std() != 0:
                    y = (y- y.mean())/y.std()
                    y_prime = (y_prime - y_prime.mean())/y_prime.std()

                else:
                    y = (y - y.mean())
                    y_prime = (y_prime - y_prime.mean())
                reward_normalized = y_prime[-1]


                new_p_euc, new_p_sr = weigh_kernels(kernel_space, kernel_temp, np.append(training_idx, [choice]), y_prime)
                marginal_euc, marginal_sr = get_ml(kernel_space, kernel_temp, np.append(training_idx, [choice]), y_prime)

                p_euc = new_p_euc

                #p_euc = (marginal_euc * p_euc) / ((marginal_euc * p_euc) + (marginal_sr * (1-p_euc)))



                #
                temp_preds = estimate_GP(kernel_temp, y, training_idx, option_indices=options)
                euc_preds = estimate_GP(kernel_space, y, training_idx, option_indices=options)
                comp_preds = estimate_GP(kernel_comp_unweighted, y, training_idx, option_indices=options)

                diff = estimate_euclidean_model(kernel, y, training_idx, option_indices=options)
                comp_rpe = comp_preds[decision] - reward_normalized

                euc_rpe = np.abs(euc_preds[decision] - reward_normalized)
                temp_rpe = np.abs(temp_preds[decision] - reward_normalized)
                w_error = euc_rpe /(euc_rpe + temp_rpe + 0.00001) # calculate how big the euclidean error is relative to the temporal
                w_euc = 1-w_error


                ### update arrays:
                context_dict[current_context]["training_idx"].append(choice)
                context_dict[current_context]["rewards"].append(reward)
                context_dict[current_context]["state_rewards"][choice] = reward

                ## store data
                data_frame[i, n] = diff
                #weights[i, n] = w_euc
                kernel = ((kernel_space*p_euc) + (kernel_temp*(1-p_euc)))
                if optimize_post_weight:
                    kernel = (kernel * w_post) + (kernel_comp_unweighted * (1- w_post))


                weights_rpe[i, n] = w_euc#np.zeros_like(data_frame)
                RPEs_euc[i, n] = euc_rpe
                RPEs_temp[i, n] = temp_rpe
                weights_ml[i, n] =  p_euc
                ml_euc[i, n] =  marginal_euc
                ml_temp[i, n] =  marginal_sr
                RPEs_comp[i, n] = comp_rpe


    data_frame = pd.DataFrame(data_frame)
    data_frame.to_csv(f"param_fits/{file_name}", header=header, index=False)

    weights_rpe = pd.DataFrame(weights_rpe)
    weights_rpe.to_csv(f"param_fits/rpe_weights_compositional.csv", header=header, index=False)

    RPEs_euc = pd.DataFrame(RPEs_euc)
    RPEs_euc.to_csv(f"param_fits/rpe_euc.csv", header=header, index=False)

    RPEs_temp = pd.DataFrame(RPEs_temp)
    RPEs_temp.to_csv(f"param_fits/rpe_temp.csv", header=header, index=False)

    weights_ml = pd.DataFrame(weights_ml)
    weights_ml.to_csv(f"param_fits/posterior_euc.csv", header=header, index=False)

    ml_euc = pd.DataFrame(ml_euc)
    ml_euc.to_csv(f"param_fits/ml_euc.csv", header=header, index=False)

    ml_temp = pd.DataFrame(ml_temp)
    ml_temp.to_csv(f"param_fits/ml_temp.csv", header=header, index=False)

    RPEs_comp = pd.DataFrame(RPEs_comp)
    RPEs_comp.to_csv(f"param_fits/rpe_comp.csv", header=header, index=False)
    return weights_rpe, RPEs_euc, RPEs_temp, weights_ml, ml_euc, ml_temp, RPEs_comp


    #return weights, RPEs_euc, RPEs_temp



with open('transitions.pickle', 'rb') as handle:
    transition_dict = pickle.load(handle)

# with open('path_integration_monster_locations_no_noise.pickle', 'rb') as handle:
#     PI_dict = pickle.load(handle)

with open('path_integration_monster_locations_no_noise_true_scale.pickle', 'rb') as handle:
    PI_dict = pickle.load(handle)

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
#
# run_compositional_sr_model(num_samples=20, file_name="comp_sr_model_pi_lowlr.csv", path_integration = True)
# run_compositional_sr_model(num_samples=20, file_name="comp_sr_model_true_loc_lowlr.csv", path_integration = False)
#run_compositional_sr_model(num_samples=20, file_name="comp_sr_model_pi_true_scale_lowlr.csv", path_integration = True)
#run_compositional_sr_model(num_samples=20, file_name="comp_sr_model_pi_true_scale_constantlambda.csv", path_integration = True, optimize_lr=True)

#run_compositional_sr_model(num_samples=20, file_name="comp_sr_model_pi_true_scale_constantlambda_multiplicative.csv", path_integration = True, optimize_lr=True)
#weights_fitted = run_compositional_sr_model(num_samples=15, file_name="comp_sr_model_rpe_weight.csv", path_integration = False, optimize_lr=True)
weights_fitted = run_compositional_sr_model(num_samples=15, file_name="comp_sr_model_bayes_weight_optimized.csv", path_integration = False, optimize_lr=False, optimize_post_weight=True)

weights_rpe, RPEs_euc, RPEs_temp, weights_ml, ml_euc, ml_temp, RPEs_comp = weights_fitted
weights_rpe, RPEs_euc, RPEs_temp, weights_ml, ml_euc, ml_temp, RPEs_comp = weights_rpe.values, RPEs_euc.values, RPEs_temp.values, weights_ml.values, ml_euc.values, ml_temp.values, RPEs_comp.values


N = 1




plt.plot(RPEs_euc[:, N].reshape(48, 100).T.mean(axis=1) - RPEs_temp[:, N].reshape(48, 100).T.mean(axis=1))
plt.plot(RPEs_euc[:, N].reshape(48, 100).T.mean(axis=1))
plt.plot(RPEs_temp[:, N].reshape(48, 100).T.mean(axis=1))
plt.plot(weights_rpe[:,N].reshape(48, 100).T.mean(axis=1))
plt.plot(weights_ml[:,N].reshape(48, 100).T.mean(axis=1))
weights_ml[:, N]
