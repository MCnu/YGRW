"""
Functions to perform a simulation of a trajectory.
"""


import numpy as np
from YGRW.trajectory import Trajectory
from YGRW.steps import Stepper, UniformSteps, FBMSteps
from tqdm import tqdm


def generate_trajectory(
    timesteps: int,
    dt: float = 0.21,
    stepper: Stepper = UniformSteps(),
    initial_position: np.ndarray = np.array((0, 0)),
    locus_radius: float = 0.01,
    bound_zone_thickness: float = 0,
    nuclear_radius: float = 1.0,
    bound_to_bound: float = 0.5,
    unbound_to_bound: float = 0.2,
    watch_progress: bool = False,
    fail_cutoff: int = 200,
    write_after: bool = False,
    write_format: str = "csv",
):
    """
    All length-scale units are in micron.

    Parameters
    ----------
    initial_position: Initial position of the locus
    timesteps: Duration of simulation
    dt: time between each step
    locus_radius: Size of particle (units of micron)
    nuc_radius: Size of nucleus (units of micron)

    Returns: Trajectory object
    -------

    """

    traj = Trajectory(
        initial_position=initial_position,
        dt=dt,
        nuclear_radius=nuclear_radius,
        locus_radius=locus_radius,
        bound_zone_thickness=bound_zone_thickness,
        bound_to_bound=bound_to_bound,
        unbound_to_bound=unbound_to_bound,
    )

    if isinstance(stepper, FBMSteps):
        assert stepper.step_batchsize == timesteps, (
            "Batch for random generation" "must agree with run length."
        )

    taken_steps = 0
    failed_steps = 0
    if watch_progress:
        pbar = tqdm(total=timesteps)

    # Central loop of Trajectory runner
    # 1. Determine if next step will be bound,
    # 2. Attempt to take next step, Rescale step depending on if bound or not,
    # 3. Adjust step if step is invalid

    while taken_steps < timesteps:

        traj.bound_states.append(
            generate_current_bound(traj, bound_to_bound, unbound_to_bound)
        )
        while failed_steps < fail_cutoff:
            if traj.is_bound:
                cur_step = stepper.generate_bound_step(
                    prev_step=traj.prev_step, prev_angle=traj.prev_angle
                )
            else:
                cur_step = stepper.generate_step(
                    prev_step=traj.prev_step, prev_angle=traj.prev_angle
                )

            if traj.check_step_is_valid(step=cur_step, is_bound=traj.is_bound):
                traj.take_step(cur_step)
                taken_steps += 1
                if watch_progress:
                    pbar.update(1)
                failed_steps = 0
                break
            else:
                cur_step = traj.step_rescale(step=cur_step, is_bound=traj.is_bound)
                # If using the FLE stepper, it must regen noise after collision
                if "FBM" in str(stepper):
                    stepper.regenerate_correlated_noise()
                # If step mod cannot take a real step, no step is taken

                rad_post_step = traj.radius_post_step(step=cur_step)

                if rad_post_step > traj.nuclear_radius:
                    cur_step = np.zeros(2)
                # Same as above, but for bound steps and bound barrier
                elif traj.is_bound and rad_post_step < (
                    traj.nuclear_radius - traj.bound_zone_thickness
                ):
                    cur_step = np.zeros(2)
                # Take newly adjusted step
                traj.take_step(cur_step)
                taken_steps += 1
                if watch_progress:
                    pbar.update(1)
                failed_steps += 1
                break

        if failed_steps == fail_cutoff:
            print(f"Warning: Run got stuck at step {taken_steps}")
            raise

    if watch_progress:
        pbar.close()

    if write_after:

        optional_header = ""
        optional_header += f"stepper:{stepper.__class__.__name__},"
        traj.write_trajectory(
            output_file=None, format=write_format, optional_header_add=optional_header
        )

    return traj


def generate_current_bound(
    traj: Trajectory,
    bound_to_bound: float = 0,
    unbound_to_bound: float = 0,
) -> bool:
    """
    Bound transition parameters describe the probability the next
    step's state is bound, and allows it to be asymmetric.

    Parameters
    ----------
    traj
    bound_to_unbound
    unbound_to_bound

    Returns
    -------
    bool: If next step will be bound or not.
    """

    # Determine if edge of locus is within the bound cutoff zone.
    cutoff_radius = traj.nuclear_radius - traj.bound_zone_thickness
    locus_extent = np.linalg.norm(traj.position) + traj.locus_radius

    in_bound_zone = cutoff_radius < locus_extent

    if not in_bound_zone:
        return False

    # If CURRENTLY bound, test if remains bound
    if traj.is_bound:
        return np.random.uniform() < bound_to_bound
    # If NOT bound, test if becomes bound
    else:
        return np.random.uniform() < unbound_to_bound
