from numba import njit, jit
import numpy as np
from math import sqrt
from tqdm import tqdm
@jit
def generate_correlated_noise(
        steps: int = None,
        gamma: float = 0.00375,
        alpha: float = 0.448,
        dt: float = 1,
        fle_random_seed: int = None,
):
    """
    Generates a series of correlated noise values.
    Based on the implementation by Yaojun Zhang in
    J.S. Lucas, Y. Zhang, O.K. Dudko, and C. Murre,
    Cell 158, 339–352, 2014
    https://doi.org/10.1016/j.cell.2014.05.036

    Which is based on the algorithm of
    C.R. Dietrich and G.N. Newsam,
    SIAM J. Sci. Comput., 18(4), 1088–1107.
    https://doi.org/10.1137/S1064827592240555

    Parameters
    ----------
    steps
    dt
    gamma
    alpha: Correlation parameter. 1 is no correlation, [1,2] is positive correlation,
            (0,1) is anticorrelation.

    Returns
    -------
    """

    # Compute correlation vector R.
    pre_r = np.zeros(shape=(steps + 1))
    pre_r[0] = 1.0
    for k in range(1, steps + 1):
        fd_addition = (
                              (k + 1) ** alpha - 2 * (k ** alpha) + (
                                  k - 1) ** alpha
                      ) / 2
        pre_r[k] = fd_addition
    nrel = len(pre_r)
    r = np.zeros(2 * nrel - 2)
    r[:nrel] = pre_r
    reverse_r = np.flip(pre_r)
    r[nrel - 1:] = reverse_r[:-1]

    # Fourier transform pre-computed values earlier
    # Corresponds to step a on page 1091 of Dietrich & Newsam,

    s = np.real(np.fft.fft(r)) / (2 * steps)
    strans = np.lib.scimath.sqrt(s)

    if fle_random_seed:
        np.random.seed(fle_random_seed)

    # Generate randomly distributed points in the complex plane (step b)
    randnorm_complex = np.random.normal(size=(2 * steps)) + 1j * np.random.normal(
        size=(2 * steps)
    )
    # Compute FFT: (step c),
    second_fft = np.fft.fft(np.multiply(strans, randnorm_complex))


    x_noise = np.real(second_fft[0:steps])
    y_noise = np.imag(second_fft[0:steps])

    return x_noise, y_noise

@njit
def step_rescale(prev_pos, step, radsquared) -> np.ndarray:
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


    a_term = step[0] ** 2 + step[1] ** 2

    b_term = 2 * (step[0] + step[1])

    c_term = (
            prev_pos[0] * prev_pos[0]
            + prev_pos[1] * prev_pos[1]
            - radsquared
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

@njit
def fast_run(noise, radius):

    N = len(noise)
    traj = np.zeros((N,2))
    traj[0] = noise[0]
    rr = radius*radius

    # build up correction vector
    for i in range(1,N):

        traj[i] = noise[i] + traj[i-1]

        extent = traj[i, 0]**2 + traj[i, 1]**2
        radius_check = extent < rr

        if not radius_check:

            cur_step = noise[i]
            cur_pos = traj[i-1]

            scaled_step = step_rescale(cur_pos, cur_step, rr)
            traj[i] = scaled_step + traj[i - 1]

    return traj

N = 1000000
alpha = .5
dt = 1e-6
gamma = 0.00375
H = alpha / 2

radius =1
scaling =sqrt(2 * gamma) * dt ** H


number_of_runs = 1000
runs = []

all_radii = []
for i in tqdm(range(number_of_runs)):

    initial_angle = np.random.uniform(0, 2*np.pi)
    initial_radius = np.random.uniform(0,1)


    steps = scaling*np.array(generate_correlated_noise(steps=N,fle_random_seed=i)).T

    steps[0][0] = np.cos(initial_angle)*initial_radius
    steps[0][1] = np.sin(initial_angle)* initial_radius
    run = fast_run(steps, 1)

    radii = np.sqrt(np.sum(np.square(run),axis=1))
    all_radii += list(radii)
    runs.append(run)






from matplotlib.pyplot import hist2d
import matplotlib.pyplot as plt

#plt.figure(figsize=(6,6))
#for run in runs:
#    plt.scatter(run[::100,0],run[::100,1],alpha=.01)

#x = np.linspace(-radius, radius, 100)
#plt.plot(x, np.sqrt(radius ** 2 - x ** 2), color="black")
#plt.plot(x, -np.sqrt(radius ** 2 - x ** 2), color="black")
#plt.show()

plt.hist(all_radii,bins=20)
plt.show()

