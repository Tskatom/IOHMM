import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix

class PoissonRegression():
    #Implements a poisson generalized linear model in python
    def __init__(self, X, Y):
        #Normalize the X values
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
        #Add a dummy variable for the intercept
        dummy = (np.matrix(np.ones(len(X)))).transpose()
        self.X = np.hstack((dummy,X))
        self.Y = np.matrix(Y).transpose()
        #Initialize beta
        self.beta = np.matrix(np.ones(self.X.shape[1]))
        self.beta = self.beta.transpose()
        self.differences = [1]

    def predict(self,X = None):
        #Creates model predictions
        if X is None:
            X = self.X
        else:
            #Normalize the X values
            for i in range(X.shape[1]):
                X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
            dummy = (np.matrix(np.ones(len(X)))).transpose()
            X = np.hstack((dummy,X))
        nu = X * self.beta
        predictions = np.exp(nu)
        predictions = np.asarray(np.squeeze(predictions))[0]
        return predictions

    def fit(self,n_iter = 100):
        #Fits the model using iteratively weighted least squares
        for i in range(n_iter):
            nu = self.X * self.beta
            mu = np.exp(nu)
            #Check for Inf values
            if np.array(np.isinf(mu)).sum() > 0:
                print "Infinite values in prediction"
                raise
            w_data = np.squeeze(np.array(mu))
            W = dia_matrix( ([w_data],0), shape=(len(w_data),len(w_data)))
            z = nu + (self.Y - mu) / (mu)
            beta_old = self.beta
            self.beta = np.linalg.pinv(self.X.transpose() * W * self.X) * self.X.transpose() * W * z
            self.differences.append(np.max((beta_old - self.beta)))

    def fit2(self,n_iter = 100):
        for i in range(n_iter):
            nu = self.X * self.beta
            mu = np.exp(nu)
            w_data = np.squeeze(np.array(mu))
            W = dia_matrix( ([w_data],0), shape=(len(w_data),len(w_data)))
            grad = self.X.T * (self.Y - mu)
            hess = -1 * self.X.T * W * self.X
            beta_old = self.beta
            self.beta = self.beta - np.linalg.pinv(hess) * grad
            self.differences.append(np.max((beta_old - self.beta)))

    def PlotConvergence(self):
        #Plots the convergence as measured by the difference in beta from the previous iteration
        plt.plot([x + 1 for x in range(len(self.differences))],self.differences)
        plt.xlabel("Iteration")
        plt.ylabel("Distance from last iteration in beta")
        plt.title("Convergence")
        plt.show()
