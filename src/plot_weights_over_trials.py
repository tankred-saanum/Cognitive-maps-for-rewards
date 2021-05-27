import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


csv = pd.read_csv("effects_and_weights.csv")
m_r = np.array(csv["m.rewards"])
r_sorted = np.argsort(m_r)
#m_r[r_sorted[-1]]
def plot_matrix():
#    avg_mat = pd.read_csv("tBt_euc_w.csv").values
    avg_mat = pd.read_csv("log_curves_weights.csv").values
    T = np.arange(len(avg_mat[0, :]))
    T = np.linspace(0, 100, len(avg_mat[0, :]))
    f, ax = plt.subplots(1, 1)
    colors = plt.cm.jet(np.linspace(0,1,len(r_sorted)))
    plt.title("Fitted logistic functions")
    plt.ylabel("Weight Euclidean")
    plt.xlabel("Trials %")
    

    for i, idx in enumerate(r_sorted):
        plt.plot(T, avg_mat[idx, :], color=colors[i], linewidth=1, alpha=0.8)

    plt.savefig("figures/may/fitted_curves.png")
    plt.show()

plot_matrix()
