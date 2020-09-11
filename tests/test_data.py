import numpy as np
import os as os

from YGRW.data_interp import JumpDistFromAngle, AngleFromAngle

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, "..", "data")


def test_open_data():
    """
    Test integrity of loading / unloading data
    Returns
    -------

    """

    isotropic_data_path = os.path.join(DATA_DIR, "jump_distances_isotropic.csv")
    data = np.loadtxt(isotropic_data_path, skiprows=1, usecols=[1, 2], delimiter=",")
    assert np.array_equal(data.shape, (512, 2))

    angle_corr_data_path = os.path.join(DATA_DIR, "angle_correlation.csv")
    data = np.loadtxt(angle_corr_data_path, skiprows=1, usecols=[1, 2], delimiter=",")
    assert np.array_equal(data.shape, (512, 2))

    angle_weighted_path = os.path.join(DATA_DIR, "jump_distances_angleweighted.csv")
    use_range = list(range(38))
    data = np.loadtxt(
        angle_weighted_path, skiprows=1, usecols=use_range[1:], delimiter=","
    )
    assert np.array_equal(data.shape, (500, 37))

    angle_weighted_path = os.path.join(DATA_DIR, "successive_angle_probability.csv")
    use_range = list(range(38))
    data = np.loadtxt(
        angle_weighted_path, skiprows=1, usecols=use_range[1:], delimiter=","
    )
    print(data.shape)
    assert np.array_equal(data.shape, (512, 37))


def test_jump_dist_from_angle():

    jdfa = JumpDistFromAngle()

    result = jdfa.distribution_from_angle(2.5)
    assert len(result) == 500
    # assert len(jdfa.jump_size_distribution_from_angle(0)) == 500

    import matplotlib.pyplot as plt

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    map = cm.get_cmap("RdBu")
    norm = Normalize(vmin=0, vmax=180)

    for i, theta in enumerate(np.arange(2.5, 180, 5)):
        data_vals = jdfa.angle_data[:, i]
        assert np.allclose(
            data_vals / np.sum(data_vals), jdfa.distribution_from_angle(theta)
        )

    # for i, theta in enumerate(np.arange(0, 180, 1)):

    #    vals = jdfa.distribution_from_angle(theta)
    #    plt.plot(jdfa.jump_values, vals/ np.sum(5*vals), alpha=.1,
    #             label=str(theta),color=map(norm(theta)))
    # plt.show()


def test_angle_from_angle():

    afa = AngleFromAngle()

    result = afa.generate_angle(2.5)
    assert len(result) == 512

    for i, theta in enumerate(np.arange(2.5, 180, 5)):

        data_vals = afa.angle_data[:, i]
        vals = afa.generate_angle(theta)
        assert np.allclose(data_vals / np.sum(data_vals), vals)
