import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
from itertools import groupby
import time
import random
from importlib import reload
from collections import defaultdict
import networkx as nx
import scipy
import pickle
import scipy.spatial


######### In this script I estimate each monster's position for each subject by path integration.


### preamble
folder = "ExplorationData"
prefix = "sub_expl_"
file_prefix = "expl"

subj_start = 101
subj_stop = 153
exclude = [136, 137, 138, 121];
subjects = list(range(subj_start, subj_stop))


## set up the perimeters of the arena
std = np.std(np.linspace(-15, 15, 100))
monster_ids = np.arange(12)
sign = [[-1, -1], [-1, 1], [1, 1], [1, -1]]

## we'll use 100 evenly spaced lengthscale values between 0.1 and 4
num_samples = 100
lengthscales = np.linspace(0.1, 4, num_samples)
estimated_euclidean_kernels = defaultdict(dict)
path_integration_kernels = defaultdict(dict)

PI_dict = {}
time_difference_dict = {}
#avg_time_differences = np.zeros((len(monster_ids), len(monster_ids)))

def RBF(X1, X2, l, sigma_f = 1):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def compute_temporal_distances(timeseries):
    unique_sorted = np.unique(timeseries)
    unique_sorted = unique_sorted.astype(int, copy=False)
    unique_sorted = unique_sorted[unique_sorted != -1]

    mat = np.zeros((len(monster_ids), len(monster_ids)))
    for i, monster in enumerate(unique_sorted):
        # if monster == -1:
        #     continue
        occurances_i = np.where(timeseries == monster)[0]
        for j, monster_j in enumerate(unique_sorted):
            if monster_j == monster:
                continue
            occurances_j = np.where(timeseries == monster_j)[0]

            # compute absolute time difference between all pairs of occurances
            temporal_dists = scipy.spatial.distance.cdist(occurances_i.reshape(-1, 1), occurances_j.reshape(-1, 1))
            min_temporal_dist = temporal_dists.min() # take the shortest time interval between monster visits
            mean_temporal_dist = temporal_dists.mean()
            mat[monster, monster_j] = min_temporal_dist
    return mat, unique_sorted # only update for monsters that were visited




for subj in subjects:
    if subj in exclude:
        continue
    runs_folder = listdir(f"{folder}/{prefix}{subj}")
    runs = range(1, len(runs_folder)+1)
    monsters = {}
    avg_time_differences = np.zeros((len(monster_ids), len(monster_ids)))
    visited_on_run_counts = np.zeros((len(monster_ids), len(monster_ids)), dtype=int)
    ## set up entries for each monster
    for m in monster_ids:
        monsters[m] = defaultdict(list)


    for run in runs:
        ## read exploration data
        df = pd.read_csv(f"{folder}/{prefix}{subj}/{file_prefix}_{run}.csv", header=None)
        df = df.values

        x_pos = df[:, 0]
        y_pos = df[:, 1]

        ## we scale by the true std
        x_pos = (x_pos - np.mean(x_pos))/std
        y_pos = (y_pos - np.mean(y_pos))/std

        closest_monster = df[:, 2]
        closest_monster -= 1


        time = np.arange(len(x_pos))
        agent_pos = np.array([x_pos[0], y_pos[0]])
        sigma = 0#0.01 ## this sigma adds noise to the path integration, set to 0 for noise free

        time_differences, update_idx = compute_temporal_distances(closest_monster)

        for u_idx1 in update_idx:
            for u_idx2 in update_idx:
                if u_idx2 == u_idx1:
                    continue
                entry = time_differences[u_idx1, u_idx2]

                avg_time_differences[u_idx1, u_idx2] += entry
                visited_on_run_counts[u_idx1, u_idx2] += 1  # normalize the time difference by this in the end.


        visit_time = 0

        for t, x, y in zip(time[1:], x_pos[1:], y_pos[1:]):

            diff = np.array([x, y]) - agent_pos  # this is the step size
            diff += np.array(sign[random.randint(0, len(sign)-1)]) * sigma  # noisy path integration
            agent_pos += diff

            current_monster = closest_monster[t]
            if current_monster > -1:

                #if current_monster not in monsters:
                current_monster = int(current_monster)

                monsters[current_monster]["x"].append(agent_pos[0])
                monsters[current_monster]["y"].append(agent_pos[1])


    Y = np.zeros((len(monsters), 2))


    for i, m in enumerate(monsters):

        if len(monsters[m]["x"]) > 0:
            monsters[m]["x"] = np.mean(monsters[m]["x"])
            monsters[m]["y"] = np.mean(monsters[m]["y"])
        else:
            print("no encounter")
            monsters[m]["x"] = 0#np.mean(monsters[m]["x"])
            monsters[m]["y"] = 0#np.mean(monsters[m]["y"])
        Y[m, 0] = monsters[m]["x"]
        Y[m, 1] = monsters[m]["y"]
     #   pos[i] = (monsters[m]["x"], monsters[m]["y"])

    PI_dict[subj] = Y
    nodes = np.arange(12)
    unvisited_idx = np.where(visited_on_run_counts == 0)
    avg_time_differences[unvisited_idx] = avg_time_differences.mean()

    visited_on_run_counts[unvisited_idx] = 1

    avg_time_differences /= visited_on_run_counts
    np.fill_diagonal(avg_time_differences, 0)
    avg_time_differences = minmax(avg_time_differences)

    time_difference_dict[subj] = avg_time_differences

    for n in range(num_samples):
        lengthscale = lengthscales[n]

        PI_K = RBF(Y, Y, l=lengthscale)

        path_integration_kernels[n][subj] = PI_K




with open('path_integration_kernels_no_noise.pickle', 'wb') as handle:
    pickle.dump(path_integration_kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('path_integration_monster_locations_no_noise.pickle', 'wb') as handle:
    pickle.dump(PI_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('avg_temporal_distances.pickle', 'wb') as handle:
#     pickle.dump(time_difference_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
