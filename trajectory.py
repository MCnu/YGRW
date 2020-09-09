""""
Describe trajectory class (contains a sequence of points that a locus travels along).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union, List
import os as os

deg = np.pi / 180


class Trajectory(object):
    def __init__(
        self,
        initial_position: np.ndarray = None,
        locus_radius: float = 0.08,
        nuclear_radius: float = 1.0,
        bound_zone_thickness: float = 0.1,
        bound_to_bound: float = None,
        unbound_to_bound: float = None
    ):

        if initial_position is None:
            initial_position = np.array([0, 0])

        self.positions = [np.array(initial_position)]
        self.bound_states = []
        self.locus_radius = locus_radius
        self.nuclear_radius = nuclear_radius
        self.bound_zone_thickness = bound_zone_thickness

        self.bound_to_bound = bound_to_bound
        self.unbound_to_bound = unbound_to_bound


    def __len__(self):
        return len(self.positions)

    @property
    def position(self):
        return self.positions[-1]

    @property
    def prev_step(self):
        if len(self.positions) < 2:
            return np.array((0, 0))
        else:
            return self.positions[-1] - self.positions[-2]

    @property
    def prev_angle(self):
        prev_step = self.prev_step

        if prev_step[0] == 0:
            sign = np.sign(prev_step[1])
            if sign == 1:
                return 90
            elif sign == -1:
                return 180
            elif sign == 0:
                return 0
        else:
            return np.arctan(prev_step[1] / prev_step[0]) * deg

    @property
    def is_bound(self):
        if len(self.bound_states) == 0:
            return False
        else:
            return self.bound_states[-1]

    def take_step(self, step: np.ndarray):
        """
        Add step to the current position and append to the position list.
        Parameters
        ----------
        step

        Returns
        -------

        """
        self.positions.append(self.position + step)

    def check_step_is_valid(self, step: np.ndarray, is_bound: bool = False) -> bool:
        """
        Return true/false if a proposed step leaves the nucleus or not.
        If the locus is bound, checks that next step stays within the bound zone.

        Parameters
        ----------
        step
        is_bound

        Returns
        -------

        """

        next_locus_extent = np.linalg.norm(self.position + step) + self.locus_radius

        # Check that locus doesn't leave bounds of the nucleus
        nuclear_check = self.nuclear_radius > next_locus_extent
        if not nuclear_check:
            return nuclear_check
        # If locus unbound, can always take next step, possibly leaving the bound zone
        if not is_bound:
            return True
        # If locus is bound, must stay in the bound zone
        diff_to_wall = self.nuclear_radius - next_locus_extent
        bound_check = diff_to_wall < self.bound_zone_thickness
        return bound_check

    def msd(self, lower_range: int = 0, upper_range: int = -1) -> float:
        """
        Returns the mean-squared displacement from the initial position.
        Returns
        -------
        msd
        """
        if upper_range == -1:
            upper_range = len(self)

        init = np.array(self.positions[0])
        sq_disps = [np.linalg.norm(pos - init) ** 2 for pos in self.positions]

        msds = []
        for i in range(lower_range + 1, upper_range + 1):
            msds.append(np.mean(sq_disps[lower_range:i]))

        return msds

    def get_avged_prev_steps(self, n_steps: int):
        """
        Return the average of the previous n_steps steps.
        Parameters
        ----------
        n_steps

        Returns
        -------

        """
        raise NotImplementedError

    def write_trajectory(
        self,
        output_file: str = None,
        format: str = "csv",
        optional_header_add: str = "",
    ):
        """
        Write an array of X/Y postion and 0 for unbound, 1 for bound state.
        Parameters
        ----------
        output_file
        format
        optional_header_add

        Returns
        -------

        """

        if output_file is None:
            cwd = os.getcwd()
            present_files = os.listdir(cwd)

            present_traj_files = [f for f in present_files if "ygrw_traj" in f]

            if len(present_traj_files) == 0:
                output_file = "ygrw_traj_" + "00000"
            else:
                cur_highest = sorted(present_traj_files)[-1]
                value = int(cur_highest[-9:-4])
                output_file = "ygrw_traj_" + str(value+1).zfill(5)

        if format == "csv" and output_file[-4:] != ".csv":
            output_file += ".csv"

        header = self.header_string() + optional_header_add
        header += "\nX,Y,is_bound\n"

        with open(output_file, "w") as f:
            f.write(header)
            for pos, bound in zip(self.positions, self.bound_states):
                f.write(f"{pos[0]},{pos[1]},{1 if bound else 0}")

    def header_string(
        self,
    ):
        the_str = ""
        the_str += f"locus_radius:{self.locus_radius},"
        the_str += f"nuclear_radius:{self.nuclear_radius},"
        the_str += f"bound_zone_thickness:{self.bound_zone_thickness},"
        the_str += f"bound_to_bound:{self.bound_to_bound},"
        the_str += f"unbound_to_bound:{self.unbound_to_bound},"

        return the_str

    def as_array(self):
        return np.array(self.positions)

    def visualize(self, vis_params=None):

        visualize_trajectory(self)

        raise NotImplementedError


def visualize_trajectory(
    traj: Union[Trajectory, List[Trajectory]], show_final_locus: bool = True
):

    plt.figure(figsize=(4, 4))

    rad = traj.nuclear_radius

    x = np.linspace(-rad, rad, 100)
    plt.plot(x, np.sqrt(rad ** 2 - x ** 2), color="black")
    plt.plot(x, -np.sqrt(rad ** 2 - x ** 2), color="black")

    positions = np.array(traj.positions)
    plt.plot(positions[:, 0], positions[:, 1], zorder=-1)

    # if traj.bound_zone_thickness:

    if show_final_locus:
        pos = traj.position
        # rad = traj.locus_radius
        plt.scatter((pos[0]), (pos[1]), color="red", s=100, marker="o", zorder=1)
        # x = np.linspace(pos[0]-rad, pos[0]+rad, 100)
        # plt.plot(x, np.sqrt(rad ** 2 - (x-pos[0]) ** 2), color="red")
        # plt.plot(x, -np.sqrt(rad ** 2 - (x-pos[0]) ** 2), color="red")

    plt.show()
