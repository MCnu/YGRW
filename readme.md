# Yeast Chromatin Random Walk Dynamics
## By Michael Chas Sumner and Steven B. Torrisi

A random walk simulator for modeling a chromatin locus diffusing through the nucleus of a yeast cell, complementary to experiments performed by Michael Sumner at the Brickner group at Northwestern University.
This is the full numerical / computational supplement to a BioRxiv preprint (*soon to come!*).

Included within this repository are the functions used to generate the random walks used in the paper,
as well as scripts which were used to generate the data which were used to fill in the paper.

### Manuscript Scripts

If you would like to review the script which was used to generate the trajectories seen in the paper,
please see the scripts folder and review `set_function.py`.

You can also run unit tests in the `tests` directory by running `pytest` to ensure that the scripts
are working on your machine.


### Simulation Library Organization
 - `steps.py` contain `Stepper` classes which govern the discrete-time random walks. A variety of step types are included,
   such as steps which draw from a uniform random distribution or a Gaussian distribution.
 - `trajectory.py` defines a `Trajectory` class which contains the full history of a chromatin loci's diffusion.
   Trajectories are generated by runs in `run.py`.
 - ` run.py` contains functions which combine `Stepper` classes to generate `Trajectory` classes.

 - `model.md` contains an overview of the model's details.
-----

#### Citation
If you use any code from this repository for your own experiments or as a starting point for the design of your own
simulator, please consider citing our manuscript:

`Random sub-diffusion and capture of genes by the nuclear pore reduces dynamics and coordinates inter-chromosomal movement, 
M.C. Sumner, S.B. Torrisi, D.G. Bricker, J. Bricker, eLife 2021;10:e66238 doi: 10.7554/eLife.66238.`


#### Acknowledgements

Helpful correspondence with Yaojun Zhang, Olga Dudko, Thomas Vojta, and Reza Vafabakhsh informed the decisions and design
of this codebase.
