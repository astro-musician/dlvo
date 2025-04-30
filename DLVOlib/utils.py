import numpy as np
import jax.numpy as jnp
import jax.random as random

def rng_key():
    return random.PRNGKey(np.random.randint(0,1e6))

def key_mcmc(n_walkers):
    return random.split(rng_key(),n_walkers)