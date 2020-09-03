import numpy as np
from YGRW.trajectory import Trajectory
import pytest


def test_initialization():

    my_traj = Trajectory()

    assert np.array_equal(my_traj.position, [0, 0])

    my_traj = Trajectory(initial_position=np.array((2, 4)))

    assert np.array_equal(my_traj.position, [2, 4])


def test_take_step():

    my_traj = Trajectory()

    step = (5, 4)

    my_traj.take_step(step=step)

    assert len(my_traj.positions) == 2
    assert np.array_equal(my_traj.position - step, my_traj.positions[0])


def test_msd():

    my_traj = Trajectory()

    assert my_traj.msd()[0] == 0

    my_traj.take_step(step=(1, 0))

    assert my_traj.msd()[0] == 0
    assert my_traj.msd()[1] == 0.5
