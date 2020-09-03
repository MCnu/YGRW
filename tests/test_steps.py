import os as os
import sys as sys

import numpy as np

from YGRW.steps import (
    Stepper,
    UniformSteps,
    GammaSteps,
    GaussianSteps,
    ExperimentalSteps,
    AngleStepper,
    UniformAngle,
    ExperimentalAngle,
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

    pass
