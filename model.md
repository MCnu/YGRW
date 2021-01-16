# Model Explanation
## Steven B. Torrisi, Michael C. Sumner

## Motivation

Experiments performed in the Brickner group at Northwestern University track
 the dynamics of chromatin loci within yeast cells as they move within
  the nuclear medium and towards the periphery of the nucleus. 
  

In order to gain a better mechanistic understanding of the dynamics of
 binding to the periphery, we built a random walk generator to capture the
  two phases.

Our simple model attempts to  recapture the observed experimental behavior via
 the following salient and physically motivated features:
 
 - *Anticorrelation.* Chromatin loci translate through the nucleoplasm while
  attached to the chromosome. This can be thought of as providing a
   small restoring force to random displacements of the locus throughout the
    nucleus. Anticorrelation between successive steps of the random walk
     captures this physical feature.
- *Limited domain.* The locus lives within the cell nucleus, which produces the
 anomalous diffusion profile expected from the MSD of unbound nucleoplasm
  motion.
 - *Two states.* The experimental portion of this study is motivated by
  understanding the dynamics of chromatin both far away and near the nuclear
   periphery, where binding to one of many nucleopore complexes (NPCs
   ) inhibits motion and plays a role in epigenetic effects. We attempt to
    model this by allowing our locus to have an internal `state` which
     describes being bound or unbound to an NPC; the binding and unbinding
      which is a stochastic
      process that is inferred through observed qualitative and quantative
       differences in loci motion in close promixmity to the periphery.



## Primary Assumptions

### Particle Size
We approximate the nucleus as a circular domain with radus $ R = 1 \mu m $
, and the chromatin locus as a circular disc with a radius $ r= 80 nm = .08
 \mu m$. The edge of the locus is not allowed to move beyond the boundaries
  of the cell (randomly sampled steps which attempt to take the locus out of
   the cell are rejected). 
### Bound and Unbound States
We model the motion of the locus as   
  ocurring in one of two phases:
  bound to the periphery of the nucleus, or unbound in the nucleoplasm.
- In the *unbound* state, the locus is free to move in the nucleoplasm.
     The locus moves in a subdiffusive regime
    , proceeding on an anticorrelated random walk with step size and angle
     distributions parameterized from experimental data.
- In the *bound* state, the locus is attached to the periphery. The
     bound state can only occur while the locus is within a `bound zone
     ` which models the periphery, defined as when the edge of the locus is
      within a
     characteristic distance $r_b$
of the nuclear wall. While bound, the locus
      is not allowed to leave the bound zone. The locus will enter the bound
       state while it is unbound in the bound zone with a random probability
        $z_{bind}$ with each
        step. While in a bound state, the locus will exit a bound state
         with a random probability $z_{unbind}$ with each step.
### Translation

We supply a random walk simulator which is uses constant timesteps and continuously varying step length.
Currently, the most sophisticated step model which we have implemented is Fractional Brownian Motion, which when the particle
is far from the boundary, can be understood as equivalent to Fractional Langevin dynamics in the overdamped limit when no driving force is present.
Please see the methods section of the manuscript (*link to come*) for greater detail on the details of the implementation.


