import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from theano import shared

# 1. generate the artificial dataset.
N = 10000

X = np.random.uniform(0, 1, N)
def DGP(x):
    obs_y = []
    for n in range(len(x)):
        if x[n] > (0.7 + np.random.normal(0.0, 0.0001, 1)[0]):
            obs_y.append(1.0)
        else:
            obs_y.append(0.0)

    return obs_y

X_shared = shared(X)
obs_y = DGP(X)

# 2. model that data with a simple regression model
with pm.Model() as exercise3:

    intercept = pm.Normal('intercept', mu=0.0, sd=.1)
    coeff = pm.Normal('beta', mu=0.0, sd=.1)

    expected_value = pm.math.invlogit((coeff * X_shared) + intercept)
    y = pm.Bernoulli('y', expected_value, observed=obs_y)

    trace = pm.sample(1000)

    pm.traceplot(trace, ['intercept', 'beta'])
    plt.show()

# 3. posterior predictive checks
TEST_N = 1000
testX = np.random.uniform(0, 1, TEST_N)
testY = DGP(testX)

X_shared.set_value(testX)

ppc = pm.sample_ppc(trace, model=exercise3, samples=500)
y_preds = ppc['y']

print "y_preds shape = ", y_preds.shape

expected_y_pred = np.reshape(np.mean(y_preds, axis=0), [-1])

plt.scatter(testX, expected_y_pred, c='g')
plt.scatter(testX, testY, c='b', alpha=0.1)
plt.title("Relationship between X and (predicted) Y")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()