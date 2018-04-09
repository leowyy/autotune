import numpy as np

def forrester(X):
    y = np.multiply(np.power(6*X-2,2), np.sin(2*(6*X-2)))
    return y