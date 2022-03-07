import time
import datetime
import numpy as np

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)