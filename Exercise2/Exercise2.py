import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# 1. generate the artificial dataset.
N = 10000

noise = np.random.normal(0.0, 0.1, N)
X = np.random.normal(1.0, 0.1, N)
obs_y = (0.65 * X) + 0.5 + noise

# 2. model that data with a simple regression model
with pm.Model() as exercise2:

    stdev = pm.HalfNormal('stdev', sd=1.)
    intercept = pm.Normal('intercept', mu=0.0, sd=1.)
    coeff = pm.Normal('beta', mu=0.5, sd=1.)

    expected_value = (X * coeff) + intercept
    y = pm.Normal('y', mu=expected_value, sd=stdev, observed=obs_y)

    trace = pm.sample(1000)

    pm.traceplot(trace, ['intercept', 'beta', 'stdev'])
    plt.show()

# 3. predictions (using training data for now)
with exercise2:

    ppc = pm.sample_ppc(trace, samples=1000) # NOTE: soon to be deprecated and replaced with sample_posterior_predictive

    y_preds = ppc['y']
    print "y_preds shape = ", ppc['y'].shape

    # Same thing, but now using the expectation
    expected_y_pred = np.reshape(np.mean(y_preds, axis=0), [-1])

    plt.scatter(X, expected_y_pred, c='g')
    plt.scatter(X, obs_y, c='b', alpha=0.1)
    plt.title("Relationship between X and (predicted) Y")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()