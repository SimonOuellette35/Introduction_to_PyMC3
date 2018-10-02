import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import pymc3 as pm
import theano
import theano.tensor as tt

# 1. generate a simple non-linear function
X = np.reshape(np.arange(-5.0, 5.0, 0.01), [-1, 1])
np.random.shuffle(X)

linearity_coeff = 0.0
Y = []
#  make this effect time-varying (relationship will "straighten out" over time)
for t in range(1000):
    linearity_coeff += 0.0002

    y_t = (linearity_coeff * X[t]) + (1 - linearity_coeff) * sp.expit(X[t])
    Y.append(y_t)

Y = np.reshape(np.array(Y), [-1])

plt.scatter(X, Y)
plt.show()

# 2. neural network
print "Building neural network..."

ann_input = theano.shared(X)
ann_output = theano.shared(Y)

n_hidden = [2, 5]
interval = 20

# Initialize random weights between each layer
init_1 = np.random.randn(X.shape[1], n_hidden[0]).astype(theano.config.floatX)
init_2 = np.random.randn(n_hidden[0], n_hidden[1]).astype(theano.config.floatX)
init_out = np.random.randn(n_hidden[1]).astype(theano.config.floatX)

with pm.Model() as neural_network:

    step_size = pm.HalfNormal('step_size', sd=np.ones(n_hidden[0]), shape=n_hidden[0])

    # Weights from input to hidden layer
    weights_1 = pm.GaussianRandomWalk('layer1', sd=step_size,
                                    shape=(interval, X.shape[1], n_hidden[0]),
                                    testval=np.tile(init_1, (interval, 1, 1)))
    weights_1_rep = tt.repeat(weights_1, ann_input.shape[0] // interval, axis=0)

    weights_2 = pm.Normal('layer2', mu=0, sd=1.,
                        shape=(1, n_hidden[0], n_hidden[1]),
                        testval=init_2)
    weights_2_rep = tt.repeat(weights_2, ann_input.shape[0], axis=0)

    weights_out = pm.Normal('layer_out', mu=0, sd=1.,
                          shape=(1, n_hidden[1]),
                          testval=init_out)
    weights_out_rep = tt.repeat(weights_out, ann_input.shape[0], axis=0)

    intercept = pm.Normal('intercept', mu=0, sd=10.)

    # Now assemble the neural network
    layer_1 = tt.tanh(tt.batched_dot(ann_input, weights_1_rep))
    layer_2 = tt.tanh(tt.batched_dot(layer_1, weights_2_rep))
    layer_out = tt.batched_dot(layer_2, weights_out_rep)

    y = pm.Normal('y', mu=layer_out + intercept, sd=0.1, observed=ann_output)

print "Done. Sampling..."

num_samples = 1000
with neural_network:

    #trace = pm.sample(num_samples, tune=num_samples, nuts_kwargs=dict(target_accept=0.98, max_treedepth=15))
    inference = pm.ADVI()
    approx = pm.fit(n=50000, method=inference)

    trace = approx.sample(draws=1000)

    pm.traceplot(trace, ['layer1', 'layer2', 'layer_out', 'intercept'])
    plt.show()

# for each of the num_samples parameter values sampled above, sample 500 times the expected y value.
samples = pm.sample_ppc(trace, model=neural_network, size=100)
y_preds = samples['y']
print "y_preds shape = ", y_preds.shape

# get the average, since we're interested in plotting the expectation.
y_preds = np.mean(y_preds, axis=1)
y_preds = np.mean(y_preds, axis=0)

RMSD = np.sqrt(np.mean((y_preds - Y) ** 2.0))

plt.scatter(X, Y)
plt.scatter(X, y_preds, alpha=0.1)
plt.show()

# show the time-varying effect
fig, axarr = plt.subplots(1, 5, sharey=True, sharex=True)

for i in range(len(axarr)):
    from_idx = 200 * i
    to_idx = 200 * (i + 1)

    axarr[i].scatter(X[from_idx:to_idx], Y[from_idx:to_idx])
    axarr[i].scatter(X[from_idx:to_idx], y_preds[from_idx:to_idx], alpha=0.1)
    axarr[i].set_title("t range %s to %s" % (from_idx, to_idx))

plt.show()