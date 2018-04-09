import pprint
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Choosing n_iter for SGDClassifier is deprecated but we need it to reproduce our result
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class MNIST_problem():
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self._initialise_data()
        self.f = lambda x: self._initialise_objective_function(x)
        self.domain = self._initialise_domain()

    def _initialise_data(self, n_train = 5000, n_test = 10000):
        # Load dataset
        mnist = fetch_mldata('MNIST original')
        X = mnist.data.astype('float64')
        X = X.reshape((X.shape[0], -1))
        y = mnist.target

        # Perform a train-test split for validation,
        # a standard scaler transforms the data in each dimension to zero mean and unit variance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size = n_train, test_size = n_test)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # for a multi-class problem:
        self.classes = np.unique(y)

        return X_train, X_test, y_train, y_test

    def _initialise_objective_function(self, x):

        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            fs[i] = 0
            gamma = np.exp(x[i,0]) 		# learning rate, log scale
            alpha = np.exp(x[i,1]) 		# l2 regulariser, log scale
            n_iter = int(x[i,2])		# num epochs
            batch_size = int(x[i,3])	# mini batch size
            clf = SGDClassifier(loss='log', penalty='l2', alpha=alpha,
                                learning_rate='constant', eta0=gamma,
                                n_iter=1)

            for j in range(n_iter):
                for (X_batch, y_batch) in self._next_batch(self.X_train, self.y_train, batch_size):
                    clf.partial_fit(X_batch, y_batch, classes=self.classes)

            score = clf.score(self.X_test, self.y_test)
            fs[i] = 1 - score # classification error
        return fs

    def _initialise_domain(self):
        domain =   [{'name': 'gamma_log','type': 'continuous', 'domain': (-6,0)},
                    {'name': 'alpha_log','type': 'continuous', 'domain': (-6,0)},
                    {'name': 'n_iter','type': 'continuous', 'domain': (5,1000)},
                    {'name': 'batch_size','type': 'continuous', 'domain': (20,2000)}]
        return domain

    # helper function for mini-batch training
    def _next_batch(self, X, y, batch_size):
        for i in np.arange(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size], y[i:i + batch_size])

    def print_domain(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.domain)