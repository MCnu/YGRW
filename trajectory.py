""""
Describe trajectory class (contains a sequence of points that a locus travels along).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Union, List
import os as os
from pylab import cm

deg = np.pi / 180


class Trajectory(object):
    def __init__(
        self,
        initial_position: np.ndarray = None,
        locus_radius: float = 0.01,
        nuclear_radius: float = 1.0,
        bound_zone_thickness: float = 0.1,
        dt: float = 0.21,
        bound_to_bound: float = None,
        unbound_to_bound: float = None,
    ):

        if initial_position is None:
            initial_position = np.array([0, 0])

        self.positions = [np.array(initial_position)]
        self.bound_states = []
        self.locus_radius = locus_radius
        self.nuclear_radius = nuclear_radius
        self.bound_zone_thickness = bound_zone_thickness
        self.dt = dt
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

    def radius_post_step(self, step: np.array, add_locus_radius: bool = True) -> float:
        """
        Compute distance from the origin for the position if it would take the
        step
        Parameters
        ----------
        step
        add_locus_radius

        Returns
        -------
        float of future position distance from origin

        """

        future_rad = np.linalg.norm(self.position + step)

        if add_locus_radius:
            future_rad += self.locus_radius

        return future_rad

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
        bool for step validity

        """

        # Determine radius if ideal step is taken
        next_locus_extent = self.radius_post_step(step=step)
        # Check that ideal position does not leave bounds of the nucleus
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

    def step_rescale(self, step: np.ndarray, is_bound: bool = False) -> np.ndarray:

        """
        When a step is determined to be invalid, this method alters the step
        to stop at the barrier, both nuclear and bound zone

        Parameters
        ----------
        step
        is_bound

        Returns step altered accordingly
        -------
        """

        next_locus_extent = self.radius_post_step(step=step)

        nuclear_check = self.nuclear_radius > next_locus_extent

        if not nuclear_check:

            a_term = step[0] ** 2 + step[1] ** 2

            b_term = 2 * (step[0] + step[1])

            c_term = (
                self.position[0] ** 2
                + self.position[1] ** 2
                - (self.nuclear_radius - 0.01) ** 2
            )

            if (4 * a_term * c_term) > (b_term ** 2):
                adj_step = step * 0.0001
                return adj_step

            lower_root = ((-b_term) - np.sqrt(b_term ** 2 - (4 * a_term * c_term))) / (
                2 * a_term
            )

            upper_root = ((-b_term) + np.sqrt(b_term ** 2 - (4 * a_term * c_term))) / (
                2 * a_term
            )

            if np.abs(upper_root) < np.abs(lower_root) and 0 < np.abs(upper_root) < 1:
                adj_step = step * (np.abs(upper_root))
            elif 0 < np.abs(lower_root) < 1:
                adj_step = step * (np.abs(lower_root))
            else:
                adj_step = step * 0.0001

            return adj_step

        diff_to_wall = self.nuclear_radius - next_locus_extent

        bound_check = diff_to_wall < self.bound_zone_thickness

        if not bound_check:

            a_term = step[0] ** 2 + step[1] ** 2

            b_term = 2 * (step[0] + step[1])

            c_term = (
                self.position[0] ** 2
                + self.position[1] ** 2
                - (self.nuclear_radius - self.bound_zone_thickness + 0.0001) ** 2
            )

            if (4 * a_term * c_term) > (b_term ** 2):
                adj_step = step * 0.0001
                return adj_step

            lower_root = ((-b_term) - np.sqrt(b_term ** 2 - (4 * a_term * c_term))) / (
                2 * a_term
            )

            upper_root = ((-b_term) + np.sqrt(b_term ** 2 - (4 * a_term * c_term))) / (
                2 * a_term
            )

            if np.abs(upper_root) < np.abs(lower_root) and 0 < np.abs(upper_root) < 1:
                adj_step = step * (np.abs(upper_root))
            elif 0 < np.abs(lower_root) < 1:
                adj_step = step * (np.abs(lower_root))
            else:
                adj_step = step * 0.0001

            return adj_step

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
                output_file = "ygrw_traj_" + str(value + 1).zfill(5)

        if format == "csv" and output_file[-4:] != ".csv":
            output_file += ".csv"

        header = self.header_string() + optional_header_add
        header += "\nX,Y,is_bound\n"

        with open(output_file, "w") as f:
            f.write(header)
            for pos, bound in zip(self.positions, self.bound_states):
                f.write(f"{pos[0]},{pos[1]},{1 if bound else 0}\n")

    def header_string(
        self,
    ):
        the_str = ""
        the_str += f"locus_radius:{self.locus_radius},"
        the_str += f"nuclear_radius:{self.nuclear_radius},"
        the_str += f"bound_zone_thickness:{self.bound_zone_thickness},"
        the_str += f"bound_to_bound:{self.bound_to_bound},"
        the_str += f"unbound_to_bound:{self.unbound_to_bound},"
        the_str += f"dt:{self.dt},"
        the_str += f"\n"

        return the_str

    def as_array(self):
        return np.array(self.positions)

    def visualize(self, vis_params=None):

        visualize_trajectory(self)


def visualize_trajectory(
    traj: Union[Trajectory, List[Trajectory]], show_final_locus: bool = True
):

    plt.figure(figsize=(4, 4))

    rad = traj.nuclear_radius

    x = np.linspace(-rad, rad, 100)
    plt.plot(x, np.sqrt(rad ** 2 - x ** 2), color="black")
    plt.plot(x, -np.sqrt(rad ** 2 - x ** 2), color="black")

    positions = np.array(traj.positions)
    bound_states = traj.bound_states
    N = len(bound_states)

    # bound_time = len(np.where(bound_states == True))

    grad_cmap = cm.get_cmap("viridis", N)

    for i, bound in zip(range(N), bound_states[:-1]):
        if not bound:
            plt.plot(
                positions[(i) : (i + 2), 0],
                positions[(i) : (i + 2), 1],
                color=colors.rgb2hex(grad_cmap(i)[:3]),
            )
        else:
            plt.plot(
                positions[i : i + 2, 0],
                positions[i : i + 2, 1],
                color="orange",
                alpha=0.5,
            )

    if traj.bound_zone_thickness:
        n, radii = 50, [
            traj.nuclear_radius - traj.bound_zone_thickness,
            traj.nuclear_radius,
        ]
        theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        xs = np.outer(radii, np.cos(theta))
        ys = np.outer(radii, np.sin(theta))

        # in order to have a closed area, the circles
        # should be traversed in opposite directions
        xs[1, :] = xs[1, ::-1]
        ys[1, :] = ys[1, ::-1]

        plt.fill(np.ravel(xs), np.ravel(ys), "gray", alpha=0.2)

    if show_final_locus:
        pos_stop = traj.position
        pos_start = traj.positions[0]

        plt.scatter(
            (pos_stop[0]), (pos_stop[1]), color="red", s=50, marker="s", zorder=N + 1
        )
        plt.scatter(
            (pos_start[0]),
            (pos_start[1]),
            color="lime",
            s=100,
            marker=">",
            zorder=N + 1,
        )

    plt.show()
