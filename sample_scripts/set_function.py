# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:28:40 2020

@author: MCS
"""
from YGRW.run import generate_trajectory
from YGRW.trajectory import Trajectory
from YGRW.steps import FBMSteps
import numpy as np

# Master random seed which dictates the generation of run-specific random seeds
np.random.seed(8675309)

# Simple trajectory parameters
nsteps = 2000  # set number of steps per trajectory
time_interval = 0.21  # set time interval between steps
nuc_rad = 1  # set nuclear radius
loc_rad = 0.001  # set locus particle radius

# translate extracted gamma from 2D to dimension-less (divide by four)
# Divide by two if extracted from 1D, divide by six if extracted from 3D

# URA3 gamma/alpha parameter inputs for steps taken in 'unbound' state
adjalpha = 0.52
adjgam = 0.015 / 4

# Parameter inputs for steps taken in 'bound' state
adjbalpha = 0.373
adjbgam = 0.003 / 4

# Bound zone parameters

bzt = 0.05  # assign bind zone thickness in micrometers, 50nm = nuclear basket height
u2b = 0.9  # assign binding rate for locus within bound zone
b2b = 0.95  # assign retention rate (aka inverse of unbinding rate)
n_trajecs = 100  # assign number of trajectories to generate

# for debugging, generate seed array for individual trajectories.
# You can thus re-run an individual trajectory if you want to inspect it in closer detail.
seed_array = np.random.uniform(10000, 99999, size=n_trajecs)

# Debug or n passage option: stop trajectory when a step cannot be completed
# for the nth time.
# This can be set to nsteps for no limit
how_big_to_fail = nsteps

# Main loop to generate trajectories
for trajecs in range(0, n_trajecs):
    # Set random seed for individual run
    cur_seed = int(seed_array[trajecs])
    print(cur_seed)
    np.random.seed(cur_seed)

    # Sets a random start position by sampling uniformly in space
    ranpos = np.random.uniform(-2.5, 2.5, size=2)
    while np.sqrt(ranpos[0] ** 2 + ranpos[1] ** 2) >= (1):
        ranpos = np.random.uniform(-1, 1, size=2)

    gtt = generate_trajectory(
        timesteps=nsteps,
        dt=time_interval,
        nuclear_radius=nuc_rad,
        locus_radius=loc_rad,
        stepper=FBMSteps(
            step_batchsize=nsteps,
            dt=time_interval,
            gamma=adjgam,
            alpha=adjalpha,
            bound_gamma=adjbgam,
            bound_alpha=adjbalpha,
        ),
        initial_position=ranpos,
        bound_to_bound=b2b,
        unbound_to_bound=u2b,
        bound_zone_thickness=bzt,
        watch_progress=True,
        fail_cutoff=how_big_to_fail,
    )
    # Uncomment method to be performed on each trajectory

    # Plot the trajectory
    Trajectory.visualize(gtt)

    # Save the trajectory in the same directory as this script
    # Trajectory.write_trajectory(gtt, output_file=f"MAMMALNUC_2p5umRAD_FLE_u90_b95_SEED8675309_{trajecs}.csv",optional_header_add="TRAJSEED={cur_seed}")
