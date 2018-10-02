import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. load the stock returns data.
series = pd.read_csv('stock_returns.csv')
returns = series.values[:1000]

# 2. first, let's see if it makes sense to fit a Gaussian distribution to this.
with pm.Model() as model1:

    stdev = pm.HalfNormal('stdev', sd=.1)
    mu = pm.Normal('mu', mu=0.0, sd=1.)

    pm.Normal('returns', mu=mu, sd=stdev, observed=returns)

    trace = pm.sample(500, tune=1000)

preds = pm.sample_ppc(trace, samples=500, model=model1)
y = np.reshape(np.mean(preds['returns'], axis=0), [-1])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(y)
ax1.set_title('Normal distribution returns')
ax2.hist(returns)
ax2.set_title('Real returns')

plt.show()

# 3. now let's relax the normal distribution assumption: let's fit a Cauchy distribution.
with pm.Model() as model2:

    beta = pm.HalfNormal('beta', sd=10.)

    pm.Cauchy('returns', alpha=0.0, beta=beta, observed=returns)

    mean_field = pm.fit(n=100000, method='advi', obj_optimizer=pm.adam(learning_rate=.001))

    trace2 = mean_field.sample(draws=10000)

preds2 = pm.sample_ppc(trace2, samples=10000, model=model2)
y2 = np.reshape(np.mean(preds2['returns'], axis=0), [-1])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(y2)
ax1.set_title('Cauchy distribution returns')
ax2.hist(returns)
ax2.set_title('Real returns')

plt.show()

print "Estimating LOO..."

# Let's compare the fit of both models
model1.name = 'Gaussian model'
model2.name = 'Cauchy model'
df_LOO = pm.compare({model1:trace, model2:trace2}, ic='LOO')

print "LOO comparison table: ", df_LOO

# 4. let's try a student t-distribution
with pm.Model() as model3:

    nu = pm.HalfNormal('nu', sd=10.)
    sigma = pm.HalfNormal('sigma', sd=.1)

    pm.StudentT('returns', nu=nu, mu=0.0, sd=sigma, observed=returns)

    mean_field = pm.fit(n=100000, method='advi', obj_optimizer=pm.adam(learning_rate=.001))

    trace3 = mean_field.sample(draws=10000)

print "Estimated nu: ", np.mean(trace3['nu'])
print "Estimated sigma: ", np.mean(trace3['sigma'])

preds3 = pm.sample_ppc(trace3, samples=1000, model=model2)
y3 = np.reshape(np.mean(preds3['returns'], axis=0), [-1])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(y3)
ax1.set_title('Student t-distribution returns')
ax2.hist(returns)
ax2.set_title('Real returns')

plt.show()

print "Estimating LOO..."

# Let's compare the fit of both models
model3.name = 'Student T model'

df_LOO = pm.compare({model2:trace2, model3:trace3}, ic='LOO')

print "LOO comparison table: ", df_LOO
