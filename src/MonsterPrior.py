import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt




class MonsterPrior():
    ''' Computes a covariance matrix useful as a GP prior based on the spatial position of the monsters'''

    def __init__(self, lengthscale = 1):
        df = pd.read_csv("monster_locations.csv", header=None)
        df = df.values
        self.pos = np.zeros((len(df[:, 0]), 2))

        # for some reason the second row is empty, we use the third row to get the y_positions of the monsters
        self.pos[:, 0] = (df[:, 0] - np.mean(df[:, 0]))/np.std(df[:, 0])
        self.pos[:, 1] = (df[:, 2] - np.mean(df[:, 2]))/np.std(df[:, 2])
        self.lengthscale = lengthscale


        self.kernel_matrix = self.__RBF(self.pos, self.pos, l = self.lengthscale)


    def __RBF(self, X1, X2, var = 1, l = 1):

        ''' Fast way of computing the RBF kernel for two coordinate matrices. See
        http://krasserm.github.io/2018/03/19/gaussian-processes/'''


        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return var**2 * np.exp(-0.5 / l**2 * sqdist)

    def plot_covariance(self):
        plt.imshow(self.kernel_matrix)
        plt.show()

    def get_kernel_matrix(self):
        return self.kernel_matrix
