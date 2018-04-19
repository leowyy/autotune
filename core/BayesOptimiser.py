"""
Deprecated, kept as a reference for comparison with Bayesian methods
"""

import numpy as np
from scipy.stats import norm
import GPy

class BayesOptimiser(object):
    def __init__(self, model, objective, X_init, Y_init):
        self.model = model
        self.objective = objective
        self.X = X_init
        self.Y = Y_init
        self.current_optimum_X, self.current_optimum_Y = self.get_current_optimum()

    def suggest_next_location_1D(self, X_test):
        acq, X_new = self.maximise_acq_fn_1D(X_test)
        return acq, X_new
    
    def process_next_location_1D(self, X_new):
        # Evaluate objective at X_new
        Y_new = self.evaluate_objective(X_new)
        
        if Y_new < self.current_optimum_Y:
            self.current_optimum_X = X_new
            self.current_optimum_Y = Y_new
        
        self.update_XY(X_new, Y_new)
        self.update_model()
        
    def evaluate_objective(self, X_new):
        return self.objective(X_new)
    
    def get_current_optimum(self):
        min_index = np.argmin(self.Y)
        return self.X[min_index], self.Y[min_index]

    def update_model(self, iters = 100):
        self.model.set_XY(self.X, self.Y)
        self.model.optimize(max_iters=iters)
    
    def update_XY(self, X_new, Y_new):
        X = np.append(self.X, X_new)
        Y = np.append(self.Y, Y_new)
        self.X = X.reshape(-1,1) 
        self.Y = Y.reshape(-1,1)
    
    def predict_with_model(self, X_test):
        mu_test, s2_test = self.model.predict(X_test)
        s_test = np.sqrt(s2_test)
        return mu_test, s_test

    def _compute_EI(self,X_test):
        '''
        Returns the expected improvement at points X_test
        '''
        Y_min = self.current_optimum_Y
        X_test = X_test.reshape(-1,1)
        Y_mu, Y_s2 = self.model.predict(X_test)
        Y_s = np.sqrt(Y_s2)
        pd = norm(0,1)
        z = (Y_min-Y_mu) / Y_s
        EI = np.multiply((Y_min-Y_mu),pd.cdf(z)) + np.multiply(Y_s,pd.pdf(z))
        assert(EI.shape[0] == X_test.shape[0])
        return EI
        
    def maximise_acq_fn_1D(self, X_test):
        '''
        Computes the acquisition function at all points on X_test
        Returns the acq fn and the x_value at the maximum
        '''
        acq_fn = lambda X: self._compute_EI(X)
        acq_values = acq_fn(X_test)
        max_index = np.argmax(acq_values)
        X_new = X_test[max_index]
        return acq_values, X_new