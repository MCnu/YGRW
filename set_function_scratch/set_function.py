# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:28:40 2020

@author: MCS
"""
from YGRW.run import generate_trajectory
from YGRW.trajectory import Trajectory
from YGRW.steps import FLESteps
import numpy as np
import math


deg = np.pi / 180

np.random.seed(57343)
nsteps = 120
time_interval = 10


# translate extracted gamma from 2D to dimension-less (divide by four)


# SPB gamma/alpha inputs
#adjgam = 0.003 / 4
#adjalpha = 0.393

# URA3 gamma/alpha inputs
adjalpha = 0.448
adjgam = (0.015/4)
adjbalpha = 0.373
adjbgam = 0.003 / 4


# assign bind zone thickness
bzt = 1.0 - math.sqrt(2 / 3)
# (1. - math.sqrt(2/3))
# assign binding rate
u2b = 0
# assign inverse of unbinding rate
b2b = 0


for trajecs in range(0, 100):
    # ranrad = 1
    ranrad = np.random.uniform(0, 1, size=1)
    radangle = np.random.uniform(low=-180, high=180, size=1)
    ranpos = np.zeros(2)
    ranpos[0] = float(np.cos(radangle * deg) * ranrad)
    ranpos[1] = float(np.sin(radangle * deg) * ranrad)
    if np.random.uniform(0, 1, size=1) > 0.5:
        ranpos[0] = -1 * ranpos[0]
    if np.random.uniform(0, 1, size=1) > 0.5:
        ranpos[1] = -1 * ranpos[1]
    #ranpos = np.array([0.99,0.])
    gtt = generate_trajectory(
        timesteps=nsteps,
        dt = time_interval,
        stepper=FLESteps(
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
        enforce_boundary = False,
    )
    Trajectory.visualize(gtt)
    #Trajectory.write_trajectory(gtt, output_file=f"URA3_20_80_FLE_{trajecs}.csv",optional_header_add="URA3_FLE_BINDING_COLLISIONrestep")
