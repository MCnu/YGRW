from YGRW.run import generate_trajectory
import numpy as np


def test_basic_run():

    traj = generate_trajectory(1)

    assert len(traj) == 2
    assert not np.array_equal(traj.position, traj.positions[0])

    impossible_traj = generate_trajectory(1, nuclear_radius=0)
    assert len(impossible_traj) == 1
