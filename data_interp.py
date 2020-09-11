import numpy as np
import os as os

FILE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(FILE_DIR, "data")


from scipy.interpolate import interp2d


class JumpDistFromAngle:
    def __init__(self, data_path: str = ""):
        if data_path is "":
            data_path = os.path.join(DATA_DIR, "jump_distances_angleweighted.csv")
        use_range = list(range(38))
        self.data = np.loadtxt(
            data_path, skiprows=1, usecols=use_range[1:], delimiter=","
        )

        self.angle_data = self.data[:, 1:]
        self.jump_values = self.data[:, 0]
        y = np.arange(2.5, 180, 5)

        self.interp = interp2d(
            x=self.jump_values, y=y, z=self.angle_data.T, kind="linear"
        )

    def distribution_from_angle(self, input_angle: float):
        """
        Bins are constructed from angles [0,5], (5,10], (10,15], ... (175,180].
        which implies bin centers at 2.5, 7.5, 12.5 ...
        So, interpolate / extrapolate from bin centers to obtain the correct
        distribution.
        """
        vals = self.interp(x=self.jump_values, y=input_angle)
        return vals / np.sum(vals)


class AngleFromAngle:
    def __init__(self, data_path: str = ""):
        if data_path is "":
            data_path = os.path.join(DATA_DIR, "successive_angle_probability.csv")
        use_range = list(range(38))
        self.data = np.loadtxt(
            data_path, skiprows=1, usecols=use_range[1:], delimiter=","
        )

        self.angle_data = self.data[:, 1:]
        self.next_angles = self.data[:, 0]
        y = np.arange(2.5, 180, 5)

        self.interp = interp2d(
            x=self.next_angles, y=y, z=self.angle_data.T, kind="linear"
        )

    def generate_angle(self, input_angle: float):
        """
        Bins are constructed from angles [0,5], (5,10], (10,15], ... (175,180].
        which implies bin centers at 2.5, 7.5, 12.5 ...
        So, interpolate / extrapolate from bin centers to obtain the correct
        distribution.
        """
        vals = self.interp(x=self.next_angles, y=input_angle)
        return vals / np.sum(vals)
