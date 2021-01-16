from YGRW.run import generate_trajectory
import numpy as np


def test_basic_run():
    """
    Verify that runs can be taken
    Returns
    -------

    """

    traj = generate_trajectory(timesteps=1, write_after=True)

    assert len(traj) == 2
    assert not np.array_equal(traj.position, traj.positions[0])

    impossible_traj = generate_trajectory(timesteps=1, nuclear_radius=0)
    assert len(impossible_traj) == 2
    assert np.array_equal(impossible_traj.positions, [[0, 0], [0, 0]])
