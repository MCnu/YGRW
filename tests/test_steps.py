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
    FBMSteps,
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


def test_fbm_stepper():

    np.random.seed(42)
    fbm_stepper = FBMSteps(step_batchsize=10)
    x1, y1 = fbm_stepper.pre_x, fbm_stepper.pre_y
    init_step = fbm_stepper.norm_msd * np.array([x1[0], y1[0]])
    returned_step = fbm_stepper.generate_step()
    assert np.array_equal(returned_step, init_step)

    np.random.seed(42)
    x2, y2 = fbm_stepper.generate_correlated_noise()
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)

    prev_x = 0
    prev_y = 0
    for n in range(10):
        x, y = fbm_stepper.generate_step()
        assert prev_x != x
        assert prev_y != y
