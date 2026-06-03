import numpy as np


class JDKFStateSpace:
    def __init__(self, X, evals, evecs, P, epsilon=None, xp=np):
        self.xp = xp
        self.evals = xp.asarray(evals)
        self.evecs = xp.asarray(evecs)

        self.epsilon = epsilon

        self.lifting_operator(X, P)
        self.state_translation()
        self.diffusion_covariance()


    def lifting_operator(self, X, P):
        '''
        This is the way they did it in the paper
        but it doesn't work because of the scalings I think
        '''
        
        N = X.shape[0]
        self.Hx = 1/(N+1) * X.T @ self.evecs

    def state_translation(self):
        Lambda = - 1/self.epsilon * self.xp.log(self.evals)
        self.A = self.xp.eye(Lambda.shape[0]) - Lambda

    def diffusion_covariance(self):
        residuals = self.evecs[1:] - self.evecs[:-1] @ self.A.T
        self.Q = self.xp.cov(residuals.T)



