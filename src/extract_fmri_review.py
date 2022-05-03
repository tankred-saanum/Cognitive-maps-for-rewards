# %% markdown
# ## Extract matrices and predictors for fMRI analyses, and value inference analyses
#
# %% codecell

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
from SuccessorRepresentation import SuccessorRepresentation
from GraphGP import LaplacianGP
import copy
import time

### Open pickled files

with open('transitions.pickle', 'rb') as handle:
    transition_dict = pickle.load(handle)


with open('path_integration_kernels.pickle', 'rb') as handle:
    estimated_euclidean_kernels = pickle.load(handle)

# with open('path_integration_monster_locations_no_noise.pickle', 'rb') as handle:
#     PI_dict = pickle.load(handle)

with open('path_integration_monster_locations_no_noise_true_scale.pickle', 'rb') as handle:
    PI_dict = pickle.load(handle)
### Unpack and process these things into a dictionary which we we'll during the analysis



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
### define the model value estimation functions,
### as well as some other helper functions
###############################################



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
    return options[0] - options[1]


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


def estimate_transition_matrix(M, gamma, lmbd = 0.0000001):

    I = np.eye(len(M))
    jitter = lmbd * np.eye(len(M))


    T = (np.linalg.inv(M + jitter) - I) / -gamma
    return T



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


def estimate_laplacian(M, gamma, lmbd=0.000001, plot=False):
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
        construct_graph_from_T(T)
    ###

    L = np.eye(len(T)) - T

    return L

def remove_outliers(array):
    idx = np.where(np.abs(array - np.mean(array)) < 2 * np.std(array))
    return idx

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


def construct_graph_from_T(T):
    A = T.copy()
    A[A>0] = 1
    g = nx.from_numpy_matrix(T)
    edges = list(g.edges)
    weights = []
    for e in edges:
        w = T[e[0], e[1]]

        # for smoother edge coloring, we scale the weights up a bit
        if w < 0.5:
            w+=0.1
        weights.append(w)



    fig, ax = plt.subplots(1, 1)
    plt.title("")
    nx.draw(g, pos, with_labels=True, edgelist = edges,  width=np.array(weights)*6)
    plt.show()
#    plt.savefig(f"figures/exploration_paths/graphs/{subj_id}.{graph_save_format}")
#    plt.show()

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


#### Unpack data

tbt_weights = pd.read_csv("tBt_euc_w_review.csv").values

# the alternative effects
effects_df = pd.read_csv("effects_and_weights_reviews.csv")
m_rewards = np.array(effects_df["m.rewards"])

final_weights_euclidean = np.array(effects_df["w.euc"])
trial_weights = np.array(effects_df["trialw"])

per_trial_df = pd.read_csv("per_trial_df_review.csv")

## note that there's a typo in "moster_rewards"
monster_rewards = pd.read_csv("moster_rewards.csv")

MR_1 = np.array(monster_rewards["ctx1"])
MR_2 = np.array(monster_rewards["ctx2"])

reward_dict = {1: MR_1, 2: MR_2}


### create matrices with predictions and RPEs

num_trials = 100
num_participants = len(np.unique(subj))

SR_GP_preds_chosen = np.zeros((num_participants, num_trials))
SR_GP_preds_unchosen = np.zeros((num_participants, num_trials))
SR_GP_RPE = np.zeros((num_participants, num_trials))

euc_GP_preds_chosen = np.zeros((num_participants, num_trials))
euc_GP_preds_unchosen = np.zeros((num_participants, num_trials))
euc_GP_RPE = np.zeros((num_participants, num_trials))

rich_euc_GP_preds_chosen = np.zeros((num_participants, num_trials))
rich_euc_GP_preds_unchosen = np.zeros((num_participants, num_trials))
rich_euc_GP_RPE = np.zeros((num_participants, num_trials))


comp_preds_chosen = np.zeros((num_participants, num_trials))
comp_preds_unchosen = np.zeros((num_participants, num_trials))
comp_RPE = np.zeros((num_participants, num_trials))

###


last_subj = -1  # make this an id so that the first participant isn't identical to this one
subj_counter = -1


### this variable controls whether the data used for fmri analysis should be saved
save_data = True
creation_date = '31.3.2022-highLengthscale'
###########



predicted_values_ctx1 = np.zeros((num_participants, 12))
predicted_values_ctx2 = np.zeros((num_participants, 12))


comp_predicted_values_ctx1 = np.zeros((num_participants, 12))
comp_predicted_values_ctx2 = np.zeros((num_participants, 12))


euc_predicted_values_ctx1 = np.zeros((num_participants, 12))
euc_predicted_values_ctx2 = np.zeros((num_participants, 12))

sr_predicted_values_ctx1 = np.zeros((num_participants, 12))
sr_predicted_values_ctx2 = np.zeros((num_participants, 12))

MT_predicted_values_ctx1 = np.zeros((num_participants, 12))
MT_predicted_values_ctx2 = np.zeros((num_participants, 12))



## initialize monster locations
mp = MonsterPrior(lengthscale=0.1)
monster_loc = mp.pos
pos = {}
for i in range(len(monster_loc)):
    pos[i] = (monster_loc[i, 0], monster_loc[i, 1])

#### set this variable to True to save simulation/graphs/control data
save_simulation_data = False
save_control = False
save_sr_graphs = False
graph_save_format ="eps"


#############################

sr_rewards = np.zeros((num_participants, num_trials))
sr_gp_rewards = np.zeros((num_participants, num_trials))
comp_rewards = np.zeros((num_participants, num_trials))
op_gp_rewards = np.zeros((num_participants, num_trials))
euc_rewards = np.zeros((num_participants, num_trials))


### here we initialize some arrays containing single predictors, which we use as control models

comp_random = np.zeros(len(subj))
mean_tracker = np.zeros(len(subj))

all_subjects = np.unique(subj)

### arrays used for computing marginal likelhoods and weights
comp_w = np.zeros(4800)  # compositional weights, estimated with log marginal likelihoods
marginal_euc = np.zeros(4800)
marginal_sr = np.zeros(4800)

### an array containing a 1 for the trials where subjects have already seen the values of both monsters:
true_diff = np.zeros(4800)
observed_both = np.zeros(4800)
# chosen_monster_visits = np.zeros(4800)
# unchosen_monster_visits = np.zeros(4800)
#

### Start loop

for i, subj_id in enumerate(subj):
    current_context = contexts[i]
    if subj_id != last_subj:

        subj_counter += 1
        trial_counter = 0
        ### set hyperparameters


        loc = PI_dict[subj_id]
        learning_rate = 0.001

        ### OPTIMAL SETTINGS FOR BEHAVIORAL MODELLING
        lengthscale_temp = 3.384
        lengthscale_spatial = 1.742

        ### LOW SETTINGS
        # lengthscale_temp = 0.1
        # lengthscale_spatial = 0.1

        ### MEDIUM SETTINGS

        # lengthscale_temp = 1
        # lengthscale_spatial = 1

        ### HIGH SETTINGS

        # lengthscale_temp = 3.5
        # lengthscale_spatial = 3.5

        context_dict = {}
        context_dict[1] ={"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))}
        context_dict[2] = {"training_idx": [], "rewards": [], "state_rewards" : np.zeros(len(np.arange(12)))}

        ### SR

        seq_list = []
        for run, seq in transition_dict[subj_id].items():
            seq_ = copy.deepcopy(seq)
            seq_ -=1
            seq_list.append(seq_)


        sr_model = SuccessorRepresentation(states, seq_list, alpha=learning_rate)
        SR = sr_model.get_SR()

        SRL = estimate_laplacian(SR, gamma = sr_model.gamma)
        kernel_temp_sr = scipy.linalg.expm(-lengthscale_temp*SRL)

        ### Transition matrix
        num_monsters = 12
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
        kernel_temp_transition = scipy.linalg.expm(-lengthscale_temp*L)

        #### now for the compositional kernel

        #kernel_temp_comp = scipy.linalg.expm(-lengthscale_temp_comp*L)


        ### Euclidean

        spatial_kernel = RBF(loc, loc, l=lengthscale_spatial)
        #spatial_kernel_comp = RBF(loc, loc, l=lengthscale_spatial_comp)



        ##### weighted
        comp_kernel = (spatial_kernel + kernel_temp_sr)/2

        if save_data:
            Path(f"fmri{creation_date}/matrices/{subj_id}").mkdir(parents=True, exist_ok=True)

            # SR_df = pd.DataFrame(T)
            # SR_df.to_csv(f"fmri{creation_date}/matrices/{subj_id}/SR_matrix.csv", index=False, header=False)

#             SR_kernel_df = pd.DataFrame(SR_kernel) ## save the temporal kernel used in the compositional model
#             SR_kernel_df.to_csv(f"fmri/matrices/{subj_id}/SR_kernel_matrix.csv", index=False, header=False)

            SR_kernel_df = pd.DataFrame(kernel_temp_sr) ## save the temporal kernel used in the compositional model
            SR_kernel_df.to_csv(f"fmri{creation_date}/matrices/{subj_id}/SR_kernel_matrix.csv", index=False, header=False)

            euclidean_kernel_df = pd.DataFrame(spatial_kernel)
            euclidean_kernel_df.to_csv(f"fmri{creation_date}/matrices/{subj_id}/euclidean_kernel_matrix.csv", index=False, header=False)

            comp_kernel_df = pd.DataFrame(comp_kernel)
            comp_kernel_df.to_csv(f"fmri{creation_date}/matrices/{subj_id}/comp_kernel_matrix.csv", index=False, header=False)



        ### add observations for this context
        options = [op1[i], op2[i]]
        choice = choices[i]
        reward = rewards[i]

        decision = decisions[i]
        unchosen = 1 - decision

        # chosen_monster_visits[i] = SR1[options[decision], options[decision]]
        # unchosen_monster_visits[i] = SR1[options[unchosen], options[unchosen]]

        true_diff_i = reward_dict[current_context][op1[i]] - reward_dict[current_context][op2[i]]
        true_diff[i] = true_diff_i




        context_dict[current_context]["training_idx"].append(choice)
        context_dict[current_context]["rewards"].append(reward)
        context_dict[current_context]["state_rewards"][choice] = reward




        SR_GP_preds_chosen[subj_counter, trial_counter] = 0
        SR_GP_preds_unchosen[subj_counter, trial_counter] = 0
        SR_GP_RPE[subj_counter, trial_counter] = (-reward)

        euc_GP_preds_chosen[subj_counter, trial_counter] = 0
        euc_GP_preds_unchosen[subj_counter, trial_counter] = 0
        euc_GP_RPE[subj_counter, trial_counter] = (-reward)


        comp_preds_chosen[subj_counter, trial_counter] = 0
        comp_preds_unchosen[subj_counter, trial_counter] = 0
        comp_RPE[subj_counter, trial_counter] = (-reward)


        ## set the last subj_id to the current one
        last_subj = subj_id
        trial_counter += 1
        comp_w[i] = 0.5


    elif len(context_dict[current_context]["rewards"]) == 0:  # check if participant has been able to make any observations in this context yet
        # if not then let choice be random, and store observations into context dict

        options = [op1[i], op2[i]]
        choice = choices[i]
        reward = rewards[i]

        decision = decisions[i]
        unchosen = 1 - decision

        # chosen_monster_visits[i] = SR1[options[decision], options[decision]]
        # unchosen_monster_visits[i] = SR1[options[unchosen], options[unchosen]]


        true_diff_i = reward_dict[current_context][op1[i]] - reward_dict[current_context][op2[i]]
        true_diff[i] = true_diff_i




        context_dict[current_context]["training_idx"].append(choice)
        context_dict[current_context]["rewards"].append(reward)
        context_dict[current_context]["state_rewards"][choice] = reward





        SR_GP_preds_chosen[subj_counter, trial_counter] = 0
        SR_GP_preds_unchosen[subj_counter, trial_counter] = 0
        SR_GP_RPE[subj_counter, trial_counter] = (-reward)

        euc_GP_preds_chosen[subj_counter, trial_counter] = 0
        euc_GP_preds_unchosen[subj_counter, trial_counter] = 0
        euc_GP_RPE[subj_counter, trial_counter] = (-reward)


        comp_preds_chosen[subj_counter, trial_counter] = 0
        comp_preds_unchosen[subj_counter, trial_counter] = 0
        comp_RPE[subj_counter, trial_counter] = (-reward)

        trial_counter += 1
        comp_w[i] = 0.5



    else:
        options = [op1[i], op2[i]]

        choice = choices[i]
        decision = decisions[i]
        unchosen = 1 - decision
        reward = rewards[i]


        true_diff_i = reward_dict[current_context][op1[i]] - reward_dict[current_context][op2[i]]
        true_diff[i] = true_diff_i

        # chosen_monster_visits[i] = SR1[options[decision], options[decision]]
        # unchosen_monster_visits[i] = SR1[options[unchosen], options[unchosen]]
        #



        training_idx = context_dict[current_context]["training_idx"] # the training indices for the gps
        R = copy.copy(context_dict[current_context]["state_rewards"]) # an array with rewards for each state for the SR. We copy so that it doesn't change when we normalize it
        y = np.array(copy.copy(context_dict[current_context]["rewards"]))  # for use in the gp models. we copy this so we can normalize it and convert it into an array without messing with the original set of reward observations

        y_prime = np.append(y, reward)

        if y.std() != 0:
            y = (y- y.mean())/y.std()
            y_prime = (y_prime - y_prime.mean())/y_prime.std()

        else:
            y = (y - y.mean())
            y_prime = (y_prime - y_prime.mean())

        reward_normalized = y_prime[-1]


        SR_GP_preds = estimate_GP(kernel_temp_sr, y, training_idx, option_indices=options)
        euclidean_preds = estimate_GP(spatial_kernel, y, training_idx, option_indices=options)
        comp_preds = estimate_GP(comp_kernel, y, training_idx, option_indices=options)


        ## alternative control models
        MT_kernel = np.eye(12)
        MT_preds = estimate_GP(MT_kernel, y, training_idx, option_indices=options)
        MT_diff = MT_preds[0] - MT_preds[1]

        mean_tracker[i] = MT_diff

        ## estimate log likelihood for SR and Euclidean
        p_euc, p_sr = weigh_kernels(spatial_kernel, kernel_temp_sr, np.append(training_idx, [choice]), y_prime)
        ml_euc, ml_sr = get_ml(spatial_kernel, kernel_temp_sr, np.append(training_idx, [choice]), y_prime)
        comp_w[i] = p_euc
        marginal_euc[i] = ml_euc
        marginal_sr[i] = ml_sr

        if (op1[i] in training_idx) and (op2[i] in training_idx):
            observed_both[i] = 1


        SR_GP_preds_chosen[subj_counter, trial_counter] = SR_GP_preds[decision]
        SR_GP_preds_unchosen[subj_counter, trial_counter] = SR_GP_preds[unchosen]
        SR_GP_RPE[subj_counter, trial_counter] = (SR_GP_preds[decision] - reward_normalized)

        euc_GP_preds_chosen[subj_counter, trial_counter] = euclidean_preds[decision]
        euc_GP_preds_unchosen[subj_counter, trial_counter] = euclidean_preds[unchosen]
        euc_GP_RPE[subj_counter, trial_counter] = (euclidean_preds[decision] - reward_normalized)

        comp_preds_chosen[subj_counter, trial_counter] = comp_preds[decision]
        comp_preds_unchosen[subj_counter, trial_counter] = comp_preds[unchosen]
        comp_RPE[subj_counter, trial_counter] = (comp_preds[decision] - reward_normalized)


        ###  for all monsters
        if trial_counter == 89 or trial_counter == 99:
            ## get unnormalized values
            y = np.array(copy.copy(context_dict[current_context]["rewards"]))

            full_comp_preds = estimate_GP_full(comp_kernel, y, training_idx)
            full_euc_preds = estimate_GP_full(spatial_kernel, y, training_idx)
            full_sr_preds = estimate_GP_full(kernel_temp_sr, y, training_idx)
            full_MT_preds = estimate_GP_full(MT_kernel, y, training_idx)

            if current_context == 1:


                predicted_values_ctx1[subj_counter] = (full_comp_preds)

                euc_predicted_values_ctx1[subj_counter] = (full_euc_preds)
                sr_predicted_values_ctx1[subj_counter] = (full_sr_preds)
                MT_predicted_values_ctx1[subj_counter] = full_MT_preds

            else:

                predicted_values_ctx2[subj_counter] = (full_comp_preds)

                euc_predicted_values_ctx2[subj_counter] = (full_euc_preds)
                sr_predicted_values_ctx2[subj_counter] = (full_sr_preds)
                MT_predicted_values_ctx2[subj_counter] = full_MT_preds



        ### update arrays:
        context_dict[current_context]["training_idx"].append(choice)
        context_dict[current_context]["rewards"].append(reward)
        context_dict[current_context]["state_rewards"][choice] = reward

        trial_counter += 1




#### Save data from control models
if save_control:
    MT_df = pd.DataFrame(mean_tracker)
    MT_df.to_csv("param_fits/mean_tracker.csv", index=False)

    true_diff_df = pd.DataFrame(true_diff)
    true_diff_df.to_csv("param_fits/true_diff.csv", index=False)
    observed_both_df = pd.DataFrame(observed_both)
    observed_both_df.to_csv("param_fits/observed_both.csv", index=False)
    # chosen_visits = pd.DataFrame(chosen_monster_visits)
    # chosen_visits.to_csv("param_fits/chosen_visits.csv", index=False)
    # unchosen_visits = pd.DataFrame(unchosen_monster_visits)
    # unchosen_visits.to_csv("param_fits/unchosen_visits.csv", index=False)




### here are some functions for estimating and smoothing the weights, as well as saving the weight data
def w_delta(w):
    w_change = np.zeros(len(w))
    for i in range(len(w) - 1):
        if i != 0:
            w_change[i] = w[i] - w[i-1]

    return w_change


def w_delta_smooth(w, n):
    w_change = np.zeros(len(w))
    for i in range(len(w) - 1):
        if i != 0:
            last_n = i-n
            if last_n < 0:
                w_change[i] = w[i] - np.mean(w[:i])

            else:
                w_change[i] = w[i] - np.mean(w[last_n:i])



    return w_change

def create_trial_data(data_list, header, path, n_rows = 100):
    ncols = len(header)
    trial_matrix = np.zeros((n_rows, ncols))
    counter = 0
    for data in data_list:
        for array in data:
            trial_matrix[:, counter] = array
            counter += 1

    trial_df = pd.DataFrame(trial_matrix)
    trial_df.to_csv(path, header=header)


### Log likelihoods
avg_ml_euc = np.mean(marginal_euc.reshape(48, 100), axis=0)
avg_ml_euc[avg_ml_euc==0] = 0.00001  ## add epsilon to avoid numerical issues
avg_ml_sr = np.mean(marginal_sr.reshape(48, 100), axis=0)
avg_ml_sr[avg_ml_sr==0] = 0.00001
nll_euc = -np.log(avg_ml_euc)
nll_sr = -np.log(avg_ml_sr)


## individual subjects
marginal_euc[marginal_euc ==0] = 0.00001
marginal_sr[marginal_sr ==0] = 0.00001
nll_euc_ind = -np.log(marginal_euc)
nll_sr_ind = - np.log(marginal_sr)


nll_list = [nll_euc, nll_sr]
nll_list_ind = [nll_euc_ind, nll_sr_ind]


### RPEs
sr_error = np.sqrt(np.mean(np.abs(SR_GP_RPE), axis=0))
euc_error = np.sqrt(np.mean(np.abs(euc_GP_RPE), axis=0))

rpe_list = [euc_error, sr_error]
rpe_list_ind = [euc_GP_RPE.ravel(), SR_GP_RPE.ravel()]

### Posterior

comp_w2 = comp_w.reshape(48, 100)
posterior = np.mean(comp_w2, axis=0)
post_list = [posterior]
post_list_ind = [comp_w]

### Predictions


avg_preds_euc = np.mean(euc_GP_preds_chosen, axis=0)
avg_preds_sr = np.mean(SR_GP_preds_chosen, axis=0)
preds_list = [avg_preds_euc, avg_preds_sr]
preds_list_ind = [euc_GP_preds_chosen.ravel(), SR_GP_preds_chosen.ravel()]

### Weights and deltas
w_avg = np.mean(tbt_weights, axis=0)
delta_w = w_delta_smooth(w_avg, 15)




delta_w_ind = np.zeros((tbt_weights.shape[0], tbt_weights.shape[1]))
for i, row in enumerate(tbt_weights):
    delta_w_i = w_delta_smooth(row, 15)
    delta_w_ind[i] = delta_w_i



######## experimental
###

# euc_evidence = marginal_euc.reshape(48, 100)
# sr_evidence = marginal_sr.reshape(48, 100)

# euc_rel_evidence = -(-np.log(euc_evidence)) - (-np.log(sr_evidence))



# idx_ = remove_outliers(np.mean(euc_rel_evidence, axis=0))
# plt.scatter(np.mean(euc_rel_evidence, axis=0)[idx_][:-1], delta_w[idx_][1:])

# scipy.stats.pearsonr(np.mean(euc_rel_evidence, axis=0)[idx_][:-1], delta_w[idx_][1:])
# plt.show()

# plt.scatter((nll_euc - nll_sr)[:-1], delta_w[1:])
# scipy.stats.pearsonr((nll_euc - nll_sr)[:-1], delta_w[1:])
# plt.show()
# delta_corrs = np.zeros(48)

# for i in range(len(delta_corrs)):
#     delta_i = delta_w_ind[i]
#     euc_ev_i = euc_ev_ind[i]
#     idx = remove_outliers(euc_ev_i)
# #    print(len(idx))
#     euc_ev_i = euc_ev_i[idx]
#     delta_i = delta_i[idx]


#     delta_corrs[i] = scipy.stats.pearsonr(euc_ev_i[1:], delta_i[:-1])[0]


# plt.hist(delta_corrs)
# plt.show()
# scipy.stats.pearsonr(delta_corrs, m_rewards)
#
####
### end experimental
###################

delta_w_ind = delta_w_ind.ravel()


w_ind = tbt_weights.ravel()


w_HP = np.array(per_trial_df["HP.euc.w"])
w_LP = np.array(per_trial_df["LP.euc.w"])
weight_list = [w_avg, w_HP, w_LP, delta_w]
weight_list_ind = [w_ind, delta_w_ind]

data_list = [nll_list, rpe_list, post_list, preds_list, weight_list]
data_list_ind = [nll_list_ind, rpe_list_ind, post_list_ind, preds_list_ind, weight_list_ind]

header = ["nll_euc", "nll_temp", "RPE_euc", "RPE_temp", "posterior_euc", "pred_euc", "pred_temp", "weights_euc", "weights_high", "weights_low", "delta_w"]

header_ind = ["nll_euc", "nll_temp", "RPE_euc", "RPE_temp", "posterior_euc", "pred_euc", "pred_temp", "weights_euc", "delta_w"]

## averaged over subjects
if save_data:
    create_trial_data(data_list, header, path= "trial_data_models_review.csv", n_rows=100)


## individual
if save_data:
    create_trial_data(data_list_ind, header_ind, path="individual_trial_data_models_review.csv", n_rows=4800)



def plot_predictions(preds, truth, title="", label=""):
    pred_mean = np.mean(preds, axis=0)
    preds_sd = np.std(preds, axis=0)
    plt.title(title)
    plt.plot(np.arange(12),pred_mean, label=label)
    plt.fill_between(np.arange(12), pred_mean + preds_sd, pred_mean - preds_sd , alpha=0.5)
    plt.plot(np.arange(12), truth, label="Ground truth")
    plt.xlabel("Monster number")
    plt.xticks(np.arange(12))
    plt.ylabel("Estimated value")
    plt.legend()
    plt.savefig(f"figures/predicted values {label} {title}.png")
    plt.show()




subjects = np.unique(subj)

def save_csv(matrix, indices, path):
    matrix_df = pd.DataFrame(matrix)
    matrix_df.index = indices
    matrix_df.to_csv(path)


if save_data:
    Path(f"fmri{creation_date}/predictions").mkdir(parents=True, exist_ok=True)
    save_csv(SR_GP_preds_chosen, subjects, f"fmri{creation_date}/predictions/SR_GP_preds_chosen.csv")
    save_csv(SR_GP_preds_unchosen, subjects, f"fmri{creation_date}/predictions/SR_GP_preds_unchosen.csv")
    save_csv(SR_GP_RPE, subjects, f"fmri{creation_date}/predictions/SR_GP_RPE.csv")

    save_csv(euc_GP_preds_chosen, subjects, f"fmri{creation_date}/predictions/RBF_preds_chosen.csv")
    save_csv(euc_GP_preds_unchosen, subjects, f"fmri{creation_date}/predictions/RBF_preds_unchosen.csv")
    save_csv(euc_GP_RPE, subjects, f"fmri{creation_date}/predictions/RBF_RPE.csv")


    save_csv(comp_preds_chosen, subjects, f"fmri{creation_date}/predictions/comp_preds_chosen.csv")
    save_csv(comp_preds_unchosen, subjects, f"fmri{creation_date}/predictions/comp_preds_unchosen.csv")
    save_csv(comp_RPE, subjects, f"fmri{creation_date}/predictions/comp_RPE.csv")

    ## compositional predictions
    save_csv(predicted_values_ctx1, subjects, f"fmri{creation_date}/predictions/comp_final_predictions1.csv")
    save_csv(predicted_values_ctx2, subjects, f"fmri{creation_date}/predictions/comp_final_predictions2.csv")

    ## euclidean predicitons
    save_csv(euc_predicted_values_ctx1, subjects, f"fmri{creation_date}/predictions/euc_final_predictions1.csv")
    save_csv(euc_predicted_values_ctx2, subjects, f"fmri{creation_date}/predictions/euc_final_predictions2.csv")

    ## temporal predictions
    save_csv(sr_predicted_values_ctx1, subjects, f"fmri{creation_date}/predictions/temporal_final_predictions1.csv")
    save_csv(sr_predicted_values_ctx2, subjects, f"fmri{creation_date}/predictions/temporal_final_predictions2.csv")

    ## MT_predictions
    save_csv(MT_predicted_values_ctx1, subjects, f"fmri{creation_date}/predictions/MT_final_predictions1.csv")
    save_csv(MT_predicted_values_ctx2, subjects, f"fmri{creation_date}/predictions/MT_final_predictions2.csv")



# %% codecell
