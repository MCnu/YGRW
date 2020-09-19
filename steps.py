import numpy as np
from scipy.stats import gengamma
import os as os
from random import sample
from abc import ABC, abstractmethod
from YGRW.data_interp import JumpDistFromAngle, AngleFromAngle

from math import sqrt

CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, "data")

deg = np.pi / 180


class Stepper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_step(self, *args, **kwargs):
        raise NotImplementedError

    def generate_bound_step(self, *args, **kwargs):
        """
        If a child class does not have this method defined,
        call child class' generate step method.
        """
        return self.generate_step(*args, **kwargs)


class AngleStepper(ABC):
    """
    Generates an angle for a successive step defined with respect to the previous
    step along [-180, 180] where clockwise is positive and counterclockwise is
    negative. In other words, an angle of 0 would correspond to no change in angle,
    +- 90 degrees correspond to right and left respectively, and -180 and 180 are
    both antiparallel to the previous angle.
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate_angle(self, *args, **kwargs):
        raise NotImplementedError


class UniformSteps(Stepper):
    def __init__(self, lower: float = -1, upper: float = 1):

        self.lower = lower
        self.upper = upper
        super().__init__()

    def generate_step(self, prev_step=None, prev_angle=None):
        return np.random.uniform(self.lower, self.upper, size=2)


class GaussianSteps(Stepper):
    def __init__(self, mu: float = 0, sig: float = 1):

        self.mu = mu
        self.sig = sig
        super().__init__()

    def generate_step(self, prev_step=None, prev_angle=None):
        return np.random.normal(loc=self.mu, scale=self.sig, size=2)

    def generate_bound_step(self, prev_step=None, prev_angle=None):
        return np.random.normal(loc=self.mu, scale=self.sig / 2, size=2)


class GaussianDragSteps(Stepper):
    def __init__(
        self,
        mu: float = 0,
        sig: float = 1,
        bound_mu: float = None,
        bound_sig: float = None,
        spring_constant: float = -0.5,
    ):
        """
        Stepper which generates steps in a uniform direction with a 'drag'
        that will apply (spring constant)*(the previous step) returning to the previous
        direction on top of a uniformly drawn angle and step size. Should be a negative
        number ranging from [-1,0] to yield an anticorrelated random walk; if positive,
        will yield a 'persistent random walk' that will cause the previous step
        to be *added* to the next step.

        Parameters
        ----------
        shape
        rate
        bound_shape
        bound_rate
        spring_constant
        """
        self.mu = mu
        self.sig = sig

        self.spring_constant = spring_constant
        self.bound_mu = bound_mu or mu
        self.bound_sig = bound_sig or sig

        super().__init__()

    def generate_step(self, prev_step=None, prev_angle=None):

        x_drag, y_drag = compute_drag(prev_step, self.spring_constant)
        x_step, y_step = np.random.normal(loc=self.mu, scale=self.sig, size=2)

        return np.array((x_step + x_drag, y_step + y_drag))

    def generate_bound_step(self, prev_step=None, prev_angle=None):

        x_drag, y_drag = compute_drag(prev_step, self.spring_constant)
        x_step, y_step = np.random.normal(loc=self.mu, scale=self.sig, size=2)

        return np.array((x_step + x_drag, y_step + y_drag))


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

    def generate_step(self, prev_step=None, prev_angle=None):

        # TODO incorporate anglestepper
        magnitude = gengamma.rvs(self.shape, self.rate, 1)
        angle = np.random.uniform(low=-180, high=180, size=1)

        x_step = np.cos(angle * deg) * magnitude
        y_step = np.sin(angle * deg) * magnitude

        return np.array((x_step, y_step))

    def generate_bound_step(self, prev_step=None, prev_angle=None):
        return gengamma.rvs(self.bound_shape, self.bound_rate, 2)


class GammaDragSteps(Stepper):
    def __init__(
        self,
        shape: float = 0,
        rate: float = 1,
        bound_shape: float = None,
        bound_rate: float = None,
        spring_constant: float = -0.5,
    ):
        """
        Stepper which generates steps in a uniform direction with a 'drag'
        that will apply (spring constant)*(the previous step) returning to the previous
        direction on top of a uniformly drawn angle and step size. Should be a negative
        number ranging from [-1,0] to yield an anticorrelated random walk; if positive,
        will yield a 'persistent random walk' that will cause the previous step
        to be *added* to the next step.

        Parameters
        ----------
        shape
        rate
        bound_shape
        bound_rate
        spring_constant
        """
        self.shape = shape
        self.rate = rate

        self.spring_constant = spring_constant
        self.bound_shape = bound_shape or shape
        self.bound_rate = bound_rate or rate

        super().__init__()

    def generate_step(self, prev_step=None, prev_angle=None):

        # TODO incorporate anglestepper
        magnitude = gengamma.rvs(self.shape, self.rate, 1)
        angle = np.random.uniform(low=-180, high=180, size=1)

        x_drag, y_drag = compute_drag(prev_step, self.spring_constant)

        x_step = np.cos(angle * deg) * magnitude + x_drag
        y_step = np.sin(angle * deg) * magnitude + y_drag

        return np.array((x_step, y_step))

    def generate_bound_step(self, prev_step=None, prev_angle=None):
        magnitude = gengamma.rvs(self.bound_shape, self.bound_rate, 1)
        angle = np.random.uniform(low=-180, high=180, size=1)

        x_drag, y_drag = self.compute_drag(prev_step, self.spring_constant)

        x_step = np.cos(angle * deg) * magnitude + x_drag
        y_step = np.sin(angle * deg) * magnitude + y_drag

        return np.array((x_step, y_step))


class UniformAngle(AngleStepper):
    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_angle():
        return np.random.uniform(low=-180, high=180, size=1)


class ExperimentalIndependentAngle(AngleStepper):
    """
    Draws from experimental observation of the distribution of angles
    in successive steps.
    """

    def __init__(self, data_path: str = None):
        data_path = data_path or os.path.join(DATA_DIR, "angle_correlation.csv")
        data = np.loadtxt(data_path, skiprows=1, delimiter=",", usecols=range(1, 3))
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.y /= np.sum(self.y)
        super().__init__()

    def generate_angle(self, prev_angle=None):

        angle = np.random.choice(self.x, size=1, p=self.y)
        sign = sample([-1, 1], k=1)
        return angle * sign


class ExperimentalCorrelatedAngle(AngleStepper):
    """
    Draws from experimental observation of the distribution of angles
    in successive steps by using the previous angle as input.
    """

    def __init__(self, data_path: str = None):

        self.afa = AngleFromAngle()

        super().__init__()

    def generate_angle(self, prev_angle):

        angle_distribution = self.afa.distribution_from_angle(prev_angle)

        next_angle = np.random.choice(
            self.afa.next_angles, size=1, p=angle_distribution
        )
        sign = sample([-1, 1], k=1)
        return next_angle * sign


class ExperimentalSteps(Stepper, UniformAngle):
    def __init__(self, data_path: str = None):

        data_path = data_path or os.path.join(DATA_DIR, "jump_distances_isotropic.csv")
        data = np.loadtxt(data_path, skiprows=1, usecols=[1, 2], delimiter=",")
        self.x = data[:, 0]
        self.y = data[:, 1]
        super(Stepper).__init__()
        super(UniformAngle).__init__()

    def generate_step(self, prev_step=None, prev_angle=None):
        """
        Generate a uniform random angle and magnitude.
        """

        angle = self.generate_angle()
        magnitude = np.random.choice(self.x, size=1, p=self.y)

        x_step = np.cos(angle * deg) * magnitude
        y_step = np.sin(angle * deg) * magnitude

        return np.array((x_step, y_step))


class GammaAngleSteps(GammaSteps):
    """
    Draws from experimental observation of the distribution of angles,
    and the distribution of magnitudes in those angles.
    """

    def __init__(
        self,
        shape: float = 0,
        rate: float = 1,
        bound_shape: float = None,
        bound_rate: float = None,
    ):

        self.astepper = ExperimentalIndependentAngle()

        super().__init__(
            shape=shape, rate=rate, bound_shape=bound_shape, bound_rate=bound_rate
        )

    def generate_step(self, prev_step: np.ndarray = None, prev_angle: float = 0):

        angle = self.astepper.generate_angle()
        new_theta = prev_angle + angle

        magnitude = gengamma.rvs(self.shape, self.rate, 1)

        x_step = np.cos(new_theta) * magnitude
        y_step = np.sin(new_theta) * magnitude
        return np.array([x_step, y_step]).reshape(2)

    def generate_bound_step(self, prev_step, prev_angle):

        angle = self.astepper.generate_angle()
        new_theta = prev_angle + angle

        angle_mag = abs(angle)
        magnitude = gengamma.rvs(self.bound_shape, self.bound_rate, 1)

        x_step = np.cos(new_theta) * magnitude
        y_step = np.sin(new_theta) * magnitude
        return np.array([x_step, y_step]).reshape(2)


class ExperimentalAngleSteps(Stepper):
    """
    Draws from experimental observation of the distribution of angles,
    and the distribution of magnitudes in those angles.
    """

    def __init__(self, astepper=ExperimentalCorrelatedAngle()):

        self.jdfa = JumpDistFromAngle()

        self.astepper = astepper

        super().__init__()

    def generate_step(self, prev_step: np.ndarray, prev_angle: float):

        angle = self.astepper.generate_angle(prev_angle)
        new_theta = prev_angle + angle

        angle_mag = abs(angle)
        jump_distribution = self.jdfa.distribution_from_angle(angle_mag)
        jump_size = np.random.choice(self.jdfa.jump_values, size=1, p=jump_distribution)

        x_step = np.cos(new_theta) * jump_size
        y_step = np.sin(new_theta) * jump_size
        return np.array([x_step, y_step]).reshape(2)

    def generate_bound_step(self, prev_step, prev_angle):

        return self.generate_step(prev_step, prev_angle) / 10


class FLESteps(Stepper):
    def __init__(
        self,
        step_batchsize: int = 200,
        gamma: float = 0.015,
        alpha: float = 0.448,
        dt: float = 0.210,
        fle_random_seed: int = None,
    ):

        self.H = alpha / 2
        self.gamma = gamma
        self.alpha = alpha
        self.dt = dt
        self.cur_step = 0
        self.step_batchsize = step_batchsize

        self.pre_x, self.pre_y = self._generate_correlated_noise(
            steps=step_batchsize, fle_random_seed=fle_random_seed
        )
        super().__init__()

    def generate_step(self, *args, **kwargs):

        if self.cur_step == self.step_batchsize:
            self.pre_x, self.pre_y = self._generate_correlated_noise(
                steps=self.step_batchsize
            )
            self.cur_step = 0

        dx = self.pre_x[self.cur_step]
        dy = self.pre_y[self.cur_step]
        self.cur_step += 1
        return np.array([dx, dy]).reshape(2)

    def generate_bound_step(self, *args, **kwargs):

        return self.generate_step(*args, **kwargs) / 10

    def _generate_correlated_noise(
        self,
        steps: int = None,
        fle_random_seed: int = None,
    ):
        """
        Generates a series of correlated noise values.
        Based on the implementation by Yaojun Zhang in
        J.S. Lucas, Y. Zhang, O.K. Dudko, and C. Murre,
        Cell 158, 339–352, 2014
        https://doi.org/10.1016/j.cell.2014.05.036

        Which is based on the algorithm of
        C.R. Dietrich and G.N. Newsam,
        SIAM J. Sci. Comput., 18(4), 1088–1107.
        https://doi.org/10.1137/S1064827592240555

        Parameters
        ----------
        steps
        dt
        gamma
        alpha: Correlation parameter. 1 is no correlation, [1,2] is positive correlation,
                (0,1) is anticorrelation.

        Returns
        -------
        """
        if steps is None:
            steps = self.step_batchsize
        # Set up prefactors to be used later
        # Compute Hurst exponent
        H = self.alpha / 2
        # Compute length scale
        norm_msd = (sqrt(2 * self.gamma)) * self.dt ** H

        # Compute correlation vector R.
        pre_r = np.zeros(shape=(steps + 1))
        pre_r[0] = 1.0
        for k in range(1, steps + 1):
            fd_addition = (
                (k + 1) ** self.alpha - 2 * (k ** self.alpha) + (k - 1) ** self.alpha
            ) / 2
            pre_r[k] = fd_addition
        nrel = len(pre_r)
        r = np.zeros(2 * nrel - 2)
        r[:nrel] = pre_r
        reverse_r = np.flip(pre_r)
        r[nrel - 1 :] = reverse_r[:-1]

        # Fourier transform pre-computed values earlier
        # Corresponds to step a on page 1091 of Dietrich & Newsam,
        s = np.real(np.fft.fft(r)) / (2 * steps)
        strans = np.lib.scimath.sqrt(s)

        if fle_random_seed:
            np.random.seed(fle_random_seed)

        # Generate randomly distributed points in the complex plane (step b)
        randnorm_complex = np.random.normal(size=(2 * steps)) + 1j * np.random.normal(
            size=(2 * steps)
        )
        # Compute FFT: (step c),
        second_fft_x = np.fft.fft(np.multiply(strans, randnorm_complex))

        randnorm_complex = np.random.normal(size=(2 * steps)) + 1j * np.random.normal(
            size=(2 * steps)
        )
        second_fft_y = np.fft.fft(np.multiply(strans, randnorm_complex))

        # Store correlated noise values. Step d.
        x_steps = norm_msd * np.real(second_fft_x[0:steps])
        y_steps = norm_msd * np.real(second_fft_y[0:steps])

        return x_steps, y_steps


def compute_drag(prev_step: None, spring_constant: float = -0.5):
    if prev_step is None:
        prev_x_drag = 0
        prev_y_drag = 0
    else:
        prev_x_drag = prev_step[0] * spring_constant
        prev_y_drag = prev_step[1] * spring_constant

    return prev_x_drag, prev_y_drag
