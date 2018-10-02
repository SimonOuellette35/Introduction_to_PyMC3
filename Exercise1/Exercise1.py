import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# 1. generate the artificial dataset. The distribution is mu=0.5, sd=0.35
obs_y = np.random.normal(0.5, 0.35, 1000)

# 2. model that data with a simple Bayesian model.
with pm.Model() as exercise1:

    stdev = pm.HalfNormal('stdev', sd=1.)
    mu = pm.Normal('mu', mu=0.0, sd=1.)

    y = pm.Normal('y', mu=mu, sd=stdev, observed=obs_y)

    trace = pm.sample(1000)

    pm.traceplot(trace, ['mu', 'stdev'])
    plt.show()
