import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3.distributions.timeseries as ts

# 1. generate 2 cointegrated assets
N = 1000
X = [50.0]
Y = [40.0]
theta = 0.02
for t in range(1, N):
    X.append(X[t-1] + np.random.normal(0.0, 0.5))
    Y.append(Y[t-1] + np.random.normal(0.0, 0.28) + theta * (0.8 * X[t-1] - Y[t-1]))

X = np.array(X)
Y = np.array(Y)

plt.plot(X)
plt.plot(Y)
plt.show()

Z = Y / X
plt.plot(Z)
plt.show()

with pm.Model() as example2:

    theta = pm.HalfNormal('theta', sd=1., testval=1.)
    coeff = pm.HalfNormal('coeff', sd=1., testval=1.)
    sigma = pm.Normal('sigma', sd=1., testval=1.)

    sde = lambda x, theta, distance: (theta * (coeff - x), sigma)
    ts.EulerMaruyama('y', 1.0, sde, [theta, coeff], shape=len(Z), testval=np.ones(len(Z)), observed=Z)

    trace = pm.sample(10000, tune=10000, nuts_kwargs=dict(target_accept=0.95))

    pm.traceplot(trace, varnames=['theta', 'coeff'])
    plt.show()

print "theta = ", np.mean(trace['theta'])
print "coeff = ", np.mean(trace['coeff'])
