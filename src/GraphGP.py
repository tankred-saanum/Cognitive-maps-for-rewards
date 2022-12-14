import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import copy
import scipy
from scipy.optimize import minimize
#from scipy import minimize

from MonsterPrior import MonsterPrior
import pickle


class LaplacianGP():
    ''' A GP model which computes the kernel function over a graph based on the graph Laplacian. However,
    you can also pass this object a covariance matrix, accompanied by a set of training indices and rewards,
    and it will use those observations to condition its predictions when calling the mean function.
    Example:
    gp = LaplacianGP()
    gp.set_training_data(training_idx, y)
    gp.set_covariance(K)
    mu = gp.mean()

    Here K is the kernel matrix for all output points

    This object also contains methods for maximizing the marginal likelihood of the data using gradient descent (scipy.optimize integration).
    This works both for the RBF kernel, as well as the diffusion kernel, if the object is given a graph Laplacian.

    '''

    def train(self, graph, observed_nodes, y, alpha = 1):


        '''
        graph: This is a networkx graph object, or something that inherits from it.
        observed_nodes: an array of integers indexing the nodes whose values were observed
        y: an array of outcome values
        alpha: the lengthscale parameter

        '''


        self.L = nx.normalized_laplacian_matrix(graph).todense()
        self.training_idx = observed_nodes

        self.y = y
        self.alpha = alpha
        self.sigma = 0.01

        self.__K(self.L, self.alpha)

    def __K(self, L, alpha):
        ''' A method which creates the 3 kernel matrices needed to compute the posterior mean and
        covariance using the exponential of the graph laplacian weighted by negative alpha. Note that
        it is assumed that the conditioning points are included in the set of evaluation points (self.K)'''

        # the full covariance matrix
        self.K = scipy.linalg.expm(-alpha * L)

        # the matrix which will contain the covariance between all training points
        self.K_obs = np.zeros((len(self.training_idx), len(self.training_idx)))
        # first get the rows of the observed points
        K_obs_rows = self.K[self.training_idx]

        # fill in with the corresponding values at the indices of the observed points
        for i, arr in enumerate(K_obs_rows):
            self.K_obs[i] = arr[self.training_idx]

        # create matrix containing covariance between all input points and all observed points
        self.K_input_obs = np.zeros((len(self.K), len(self.training_idx)))


        # fill in with the values of indices of observations
        for i in range(len(self.K)):

            self.K_input_obs[i] = self.K[i][self.training_idx]


    def mean(self, sigma=0.01, jitter = 0.0000001):
        ''' computes the posterior mean function '''

        self.inv_K = np.linalg.inv(self.K_obs + (sigma*np.eye(len(self.K_obs))))

        return self.K_input_obs @ (self.inv_K) @ self.y

    def covariance(self, sigma = 0.1):
        ''' computes the posterior covariance '''

        return self.K - (self.K_input_obs @ np.linalg.inv(self.K_obs + sigma * np.eye(len(self.K_obs))) @ self.K_input_obs.T)


    def get_prior_covariance(self):
        ''' Getter for the kernel matrix'''
        return self.K

    def set_training_data(self, training_idx, y):
        ''' Set training data for the GP'''
        self.training_idx = training_idx
        self.y = y

    def set_covariance(self, covariance_matrix):

        ''' This method allows one to set the full covariance matrix needed to arbitrary matrices
        (i.e. the matrix isn't computed from the graph Laplacian). This is useful if the covariance
        one wishes to use is already known for instance'''

        self.K = covariance_matrix

        # the matrix which will contain the covariance between all training points
        self.K_obs = np.zeros((len(self.training_idx), len(self.training_idx)))
        # first get the rows of the observed points
        K_obs_rows = self.K[self.training_idx]

        # fill in with the corresponding values at the indices of the observed points
        for i, arr in enumerate(K_obs_rows):
            self.K_obs[i] = arr[self.training_idx]

        self.K_input_obs = np.zeros((len(self.K), len(self.training_idx)))
        # fill in with the values of indices of observations
        for i in range(len(self.K)):
            self.K_input_obs[i] = self.K[i][self.training_idx]



    def RBF(self, X1, X2, var = 1, l = 1):
        ''' Computes the RBF similarity between two n x m matrices, where n is
        the number of observations, and m is the number of feature dimensions'''

        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return var**2 * np.exp(-0.5 / l**2 * sqdist)

    def assign_inputs(self, X):
        '''Convenience function for nll minimization'''
        if len(list(X.shape)) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = X

    def nll(self, theta):
        ''' This function is adapted from Martin Krasser's tutorial on GP regression,
        using a Cholesky decomposition as a more numerically stable method for getting
        the negative log likelihood, introduced in Rasmussen and Williams'''
        l = theta[0]
        noise = theta[1]
        K = self.RBF(self.X, self.X, var=noise, l=l)
        K = K + ((noise**2) *np.eye(len(self.y)))


        L = np.linalg.cholesky(K)

        S1 = scipy.linalg.solve_triangular(L, self.y, lower=True)
        S2 = scipy.linalg.solve_triangular(L.T, self.y, lower=False)

        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * self.y.dot(S2) + \
               0.5 * len(self.training_idx) * np.log(2*np.pi)


    def set_laplacian_matrix(self, L):
        self.L = L


    def nll_diffusion_kernel(self, theta):
        ''' Performs nll minimization with scipy on a diffusion kernel'''
        l = theta[0]
        noise = 0.01  ## add jitter
        self.__K(self.L, l)

        K_ = self.K_obs.copy()
        K_ = K_ + ((noise**2)*np.eye(len(self.y)))

        try:
            L = np.linalg.cholesky(K_)
#            L = scipy.linalg.cholesky(K_)
        except np.linalg.LinAlgError as err:
            print("Warning: Cholesky didn't work - trying to remove negative eigenvalues and reconstruct using Eigendecomposition")


#            print(l)
            eig_v, eig_vec = np.linalg.eig(K_)
            eig_v[eig_v < 0] = -eig_v[eig_v < 0]
            lam = np.eye(len(K_))
            np.fill_diagonal(lam, eig_v)

            K_ = eig_vec @ lam @ np.linalg.inv(eig_vec + (np.eye(len(eig_vec))*0.000000001))
            try:
                L = np.linalg.cholesky(K_)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("Could not compute Cholesky decomposition after removing negative eigenvalues")


        S1 = scipy.linalg.solve_triangular(L, self.y, lower=True)
        S2 = scipy.linalg.solve_triangular(L.T, self.y, lower=False)

        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * self.y.dot(S2) + \
               0.5 * len(self.training_idx) * np.log(2*np.pi)




    def evaluate_nll(self, noise=0.01):
        ''' This one is better suited if you just want the nll of the GP's kernel kernel.
        Assuming 0 noise'''

        K_ = self.K_obs.copy()
        K_ += ((noise**2)*np.eye(len(self.y)))
        L = np.linalg.cholesky(K_)

        S1 = scipy.linalg.solve_triangular(L, self.y, lower=True)
        S2 = scipy.linalg.solve_triangular(L.T, self.y, lower=False)

        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * self.y.dot(S2) + \
               0.5 * len(self.training_idx) * np.log(2*np.pi)

    def minimize_nll(self, X, X_train):
        ''' Minimize nll function to be called when the kernel is RBF'''
        self.assign_inputs(X_train)

        l = np.random.uniform(0.01, 4)
        n = np.random.uniform(0.0001, 1)
        output = minimize(self.nll, [l, n], bounds=((1e-5, None), (1e-5, None)),
                          method='L-BFGS-B')
        l, n = output.x
        if len(list(X.shape)) == 1:
            X = X.reshape(-1, 1)
        else:
            X = X

        return self.RBF(X, X, var=n, l=l), l, n

    def minimize_nll_diffusion(self):
        ''' Minimize nll function to be called when the kernel is a diffusion kernel'''

        l = np.random.uniform(0.01, 4)

        try:
            output = minimize(self.nll_diffusion_kernel, [l], bounds=((1e-5, None), ),
                          method='L-BFGS-B')
        except np.linalg.LinAlgError:
            print("Could not compute cholesky - lengthscale is set to 1")
            return 1
        l = output.x

        return l
