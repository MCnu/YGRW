import numpy as np


class Stepper(object):
    def __init__(self):
        pass

    def generate_step(self):
        raise NotImplementedError

    def generate_bound_step(self):
        """
        If a child class does not have this method defined,
        call child class' generate step method.
        """
        return self.generate_step()


class UniformSteps(Stepper):
    def __init__(self, lower: float = -1, upper: float = 1):

        self.lower = lower
        self.upper = upper
        super().__init__()

    def generate_step(self):
        return np.random.uniform(self.lower, self.upper, size=2)


class GaussianSteps(Stepper):
    def __init__(self, mu: float = 0, sig: float = 1):

        self.mu = mu
        self.sig = sig
        super().__init__()

    def generate_step(self):
        return np.random.normal(loc=self.mu, scale=self.sig, size=2)

    def generate_bound_step(self):
        return np.random.normal(loc=self.mu, scale=self.sig / 2, size=2)
