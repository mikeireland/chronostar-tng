"""
This comes directly from:
https://tcrundall.github.io/chronostar-tng/examples/workedexamples.html
"""

import numpy as np
from chronostar import synthdata
from chronostar import datatools

# Define a "background" component
DIM = 6
bg_mean = np.zeros(DIM)
bg_stdev_pos = 500.
bg_stdev_vel = 30.
bg_cov = np.eye(DIM)
bg_cov[:3] *= bg_stdev_pos**2
bg_cov[3:] *= bg_stdev_vel**2

bg_age = 0.
bg_nstars = 1_000

# Define an "association" component
assoc_mean = np.ones(DIM)
assoc_stdev_pos = 50.
assoc_stdev_vel = 1.5
assoc_age = 30.
assoc_nstars = 200

assoc_cov = np.eye(DIM)
assoc_cov[:3] *= assoc_stdev_pos**2
assoc_cov[3:] *= assoc_stdev_vel**2

# Generate each component stars' current positions in cartesian space
seed = 0
rng = np.random.default_rng(seed)

# Generate a component with current day mean `bg_mean`, and with a birth covariance
# of `bg_cov`
bg_stars = synthdata.generate_association(
    bg_mean, bg_cov, bg_age, bg_nstars, rng=rng,
)
assoc_stars = synthdata.generate_association(
    assoc_mean, assoc_cov, assoc_age, assoc_nstars, rng=rng,
)
stars = np.vstack((assoc_stars, bg_stars))

# To make things easy, we scale down typical Gaia uncertainties by 70%
synthdata.SynthData.m_err = 0.3

# Measure astrometry and get a fits table, this will be the input into `prepare-data`
astrometry = synthdata.SynthData.measure_astrometry(stars)

# Lets just quickly check for bad data, in particular, bad parallaxes
# we know the stars have a position spread of 1000 pc, so shouldn't be
# many stars beyond 1,000 pc, therefore we ignore any stars with parallax
# less than 1 mas
subset_astrometry = astrometry[np.where(astrometry['parallax'] > 1.)]
subset_astrometry.write('synth_astro.fits', overwrite=True)

# Just for fun, lets save the "true memberships", but mask out stars with bad
# parallax
true_membs = np.zeros((assoc_nstars + bg_nstars, 2))
true_membs[:assoc_nstars, 0] = 1.
true_membs[assoc_nstars:, 1] = 1.
subset_true_membs = np.copy(true_membs[np.where(astrometry['parallax'] > 1.)])
np.save('true_membs.npy', subset_true_membs)

# And if you want the data in arrays for whatever reason...
astro_data = datatools.extract_array_from_table(astrometry)
astro_means, astro_covs = datatools.construct_covs_from_data(astro_data)