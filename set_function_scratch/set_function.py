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

#for trig functions
deg = np.pi / 180

#set seed for reproducibility!
np.random.seed(57343)
#howmany steps in the trajectory
nsteps = 3000
#the delta tau between each step
time_btwn_steps = 0.5

# translate extracted gamma from 2D to dimension-less (divide by four)

# SPB gamma/alpha inputs
#adjgam = 0.003 / 4
#adjalpha = 0.393

# URA3 gamma/alpha inputs
adjalpha = 0.448
adjgam = (0.015/4)

#Bound state gamma/alpha inputs
adjbalpha = 0.373
adjbgam = 0.003 / 4


# assign bind zone thickness (outer third of area = 1.0 - math.sqrt(2 / 3))
bzt = 1.0 - math.sqrt(2 / 3)

# assign binding rate
u2b = 0
# assign inverse of unbinding rate
b2b = 0



for trajecs in range(0, 100):
    #ranrad = 1
    #ranrad = np.random.uniform(0, 1, size=1)
    #radangle = np.random.uniform(low=-180, high=180, size=1)
    #ranpos = np.zeros(2)
    #ranpos[0] = float(np.cos(radangle * deg) * ranrad)
    #ranpos[1] = float(np.sin(radangle * deg) * ranrad)
    ranpos = np.random.uniform(-1,1,size = 2)
    radcheck = np.sqrt(ranpos[0]**2 + ranpos[1]**2) 
    while radcheck > 1:
        ranpos = np.random.uniform(-1,1,size = 2)
        radcheck = np.sqrt(ranpos[0]**2 + ranpos[1]**2) 
    gtt = generate_trajectory(
        timesteps=nsteps,
        stepper=FLESteps(
            step_batchsize=nsteps,
            gamma=adjgam,
            alpha=adjalpha,
            bound_gamma=adjbgam,
            bound_alpha=adjbalpha,
        ),
        dt = time_btwn_steps,
        initial_position=ranpos,
        bound_to_bound=b2b,
        unbound_to_bound=u2b,
        bound_zone_thickness=bzt,
        watch_progress=True,
    )
    #Trajectory.visualize(gtt)
    Trajectory.write_trajectory(gtt, output_file=f"URA3_REPOS_00_00_FLE_{trajecs}.csv", optional_header_add="URA3_FLE_BINDING_COLLISIONrestep")
