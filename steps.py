import numpy as np
from scipy.stats import gengamma
import os as os
from random import sample

CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, "data")


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


class GammaSteps(Stepper):
    def __init__(
        self,
        shape: float = 0,
        rate: float = 1,
        bound_shape: float = None,
        bound_rate: float = None,
    ):
        self.shape = shape
        self.rate = rate

        self.bound_shape = bound_shape or shape
        self.bound_rate = bound_rate or rate

        super().__init__()

    def generate_step(self):
        return gengamma.rvs(self.shape, self.rate, 1)

    def generate_bound_step(self):
        return gengamma.rvs(self.bound_shape, self.bound_rate, 1)



class AngleStepper(object):
    def __init__(self):
        pass

    def generate_angle(self):
        raise NotImplementedError

class UniformAngle(object):
    def __init__(self):
        pass

    def generate_angle(self):

        return np.random.uniform(low=-180, high=180, size=1)


class ExperimentalAngle(object):
    def __init__(self):

        data_path = os.path.join(DATA_DIR, 'angle_correlation')
        data = np.loadtxt(data_path, skiprows=1, delimiter=',')
        self.x = data[:, 1]
        self.y = data[:, 2]


    def generate_angle(self):

        angle = np.random.choice(self.x, size=1, p=self.y)
        sign = sample([-1, 1], k=1)
        return angle * sign