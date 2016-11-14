import numpy as np
import random

class GillespiePrior(object):

    def __init__(self,n_samples,parameter_bounds):
        self.n_samples = n_samples
        self.bounds = parameter_bounds

    def sample(self):
        result = []
        for i in range(self.n_samples):
            result.append([random.uniform(a = bounds[0], b=bounds[1]) for bounds in self.bounds])
        return result
