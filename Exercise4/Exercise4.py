import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from theano import shared

# 1. generate the artificial dataset.
N = 100
NUM_FACTORS = 15
N_PER_FACTOR = N / NUM_FACTORS

COMMON_BETA_MU = 5.6
COMMON_ALPHA_MU = 1.0
COMMON_SD = 0.1

factors = []
for _ in range(NUM_FACTORS):
    Bmu = np.random.normal(COMMON_BETA_MU, COMMON_SD)
    Amu = np.random.normal(COMMON_ALPHA_MU, COMMON_SD)

    factors.append([Amu, Bmu])

Y = None
X = None
factor_indices = None

for f_idx, f in enumerate(factors):
    innovation = np.reshape(np.random.normal(0.0, 0.01, N_PER_FACTOR), [-1, 1])

    indices = np.ones_like(innovation) * f_idx
    y = (f[1] * innovation) + f[0]

    if X is None:
        X = innovation
        factor_indices = indices
        Y = y
    else:
        X = np.concatenate((X, innovation), axis=0)
        factor_indices = np.concatenate((factor_indices, indices), axis=0)
        Y = np.concatenate((Y, y), axis=0)

X_shared = shared(X)
Y_shared = shared(Y)

# 2. first produce the non-hierarchical model, independent regression for each factor
with pm.Model() as exercise4_unpooled:

    a = pm.Normal('a', mu=1, sd=.5, shape=NUM_FACTORS)
    b = pm.Normal('b', mu=5, sd=.5, shape=NUM_FACTORS)

    expected_value = (b[factor_indices.astype(int)] * X_shared) + a[factor_indices.astype(int)]
    y = pm.Normal('y', expected_value, observed=Y_shared)

    trace = pm.sample(2000, tune=2000)

    pm.traceplot(trace, ['a', 'b'])
    plt.show()

# evaluate model RMSD
samples = pm.sample_ppc(trace, model=exercise4_unpooled, size=1000)
y_preds = np.mean(samples['y'], axis=1)

print "y_preds shape = ", y_preds.shape
print "Y shape = ", Y.shape

# print "y_preds = ", y_preds[:10]
# print "Y = ", Y[:10]

RMSD = np.sqrt(np.mean((y_preds - Y) ** 2.0))
print "RMSD of unpooled model: ", RMSD

# 3. now produce the hierarchical version with a shared factor
with pm.Model() as exercise4_pooled:

    mu_a = pm.Normal('mu_a', 1, sd=10, shape=1)
    sigma_a = pm.HalfNormal('sigma_a', sd=.5)
    mu_b = pm.Normal('mu_b', 5, sd=10, shape=1)
    sigma_b = pm.HalfNormal('sigma_b', sd=.5)

    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=NUM_FACTORS)
    b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=NUM_FACTORS)

    expected_value = (b[factor_indices.astype(int)] * X_shared) + a[factor_indices.astype(int)]
    y = pm.Normal('y', expected_value, observed=Y_shared)

    trace = pm.sample(2000, tune=2000)

    pm.traceplot(trace, ['mu_a', 'sigma_a', 'mu_b', 'sigma_b', 'a', 'b'])
    plt.show()

# evaluate model accuracy
samples = pm.sample_ppc(trace, model=exercise4_pooled, size=1000)
y_preds = np.mean(samples['y'], axis=1)

RMSD = np.sqrt(np.mean((y_preds - Y) ** 2.0))
print "RMSD of hierarchical model: ", RMSD
