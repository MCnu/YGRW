import os as os
import sys as sys

import numpy as np
from collections import Counter

CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, "../data")

from YGRW.steps import (
    Stepper,
    UniformSteps,
    GammaSteps,
    GaussianSteps,
    ExperimentalSteps,
    AngleStepper,
    UniformAngle,
    ExperimentalIndependentAngle,
)


def test_basic_init():

    assert isinstance(UniformSteps(), Stepper)
    assert isinstance(GammaSteps(), Stepper)
    assert isinstance(UniformAngle(), AngleStepper)
    assert isinstance(ExperimentalSteps(), Stepper)


def test_uniform_stepper():

    stepper = UniformSteps()
    assert len(stepper.generate_step()) == 2


def test_gaussian_stepper():

    stepper = GaussianSteps(mu=3, sig=0)
    step = stepper.generate_step()
    assert len(stepper.generate_step()) == 2
    assert np.array_equal(step, [3.0, 3.0])

    stepper = GaussianSteps(mu=0, sig=1)
    step = stepper.generate_step()
    assert len(stepper.generate_step()) == 2
    assert not np.array_equal(step, [3.0, 3.0])


def test_gamma_stepper():

    pass


def test_experimental_angle():

    data_path = os.path.join(DATA_DIR, "angle_correlation.csv")
    data = np.loadtxt(data_path, skiprows=1, delimiter=",", usecols=range(1, 3))

    X = data[:, 0]
    Y = data[:, 1]

    n_samples = 5000

    astepper = ExperimentalIndependentAngle()

    angles = [float(abs(astepper.generate_angle())) for _ in range(n_samples)]

    assert np.mean(angles) > 90
