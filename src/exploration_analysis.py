import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import pickle
from itertools import groupby
import time

### Here I extract, analyse and save subjects exploration data so it's easy to access
### later when we want to fit models using this

folder = "ExplorationData"
prefix = "sub_expl_"
file_prefix = "expl"

subj_start = 101
subj_stop = 153

subjects = list(range(subj_start, subj_stop))
runs = list(range(1, 5))

state_occupancy_dict = {}
occupancy_histograms = {}


for subj in subjects:
    state_occupancy_dict[subj] = {}
    occupancy_histograms[subj] = {}
    total_monster_counts = np.zeros(12) # there are 12 monsters, we wish to maintain a count of how  often  individual participants encountered them
    
    for run in runs:
        monster_counts = np.zeros(12)
        df = pd.read_csv(f"{folder}/{prefix}{subj}/{file_prefix}_{run}.csv", header=None)
        df = df.values

        x_pos = df[:, 0]
        y_pos = df[:, 1]
        closest_monster = df[:, 2]
        t = np.linspace(0, 1, len(x_pos))

        ### display paths
        # fig, ax = plt.subplots(1, 1)
        # plt.title(f"Subject id: {subj}")
        # ax.plot(x_pos, y_pos)
        # plt.savefig(f"figures/exploration_paths/{subj}")
#        plt.show()


        ### First we compute the compressed state occupancy sequence (we remove adjacent duplicates)
        
        #remove zeroes
        closest_monster = closest_monster[closest_monster != 0].astype(int)
        # use the groupby function from itertools
        state_occupancy = np.array([monsters[0] for monsters in groupby(closest_monster)])
        # save occupancy array in the dictionary
        state_occupancy_dict[subj][run] = state_occupancy

        # now we compute their histograms (I prefer using np.uniqe rather np.hist, because we can
        # control what the "bins" represent, but np.hist should also work).

        unique, idx, counts = np.unique(closest_monster, return_index=True, return_counts=True)
        unique -= 1  # subtract 1 so the monster number becomes an index in our array
        monster_counts[unique] = counts  # add the counts to those indices
        total_monster_counts += monster_counts

        occupancy_histograms[subj][run] = monster_counts
    occupancy_histograms[subj]["total"] = total_monster_counts

pickle the transitions stored in the dictionary for later use
with open('transitions.pickle', 'wb') as handle:
    pickle.dump(state_occupancy_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('occupancy_counts.pickle', 'wb') as handle:
#     pickle.dump(occupancy_histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)





