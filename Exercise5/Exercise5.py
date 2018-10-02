import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import pymc3 as pm
import theano

# 1. generate a simple non-linear function
X = np.reshape(np.arange(-5.0, 5.0, 0.01), [-1, 1])

print "shape of X: ", X.shape

Y = X ** 2.0

plt.scatter(X, Y)
plt.show()

# 2. neural network
print "Building neural network..."

ann_input = theano.shared(X)
ann_output = theano.shared(Y)

n_hidden = 5

# Initialize random weights between each layer
init_1 = np.random.randn(X.shape[1], n_hidden).astype(theano.config.floatX)
init_2 = np.random.randn(n_hidden, n_hidden).astype(theano.config.floatX)
init_out = np.random.randn(n_hidden, 1).astype(theano.config.floatX)

SD = 10.
with pm.Model() as neural_network:

    # Weights from input to hidden layer
    weights_1 = pm.Normal('layer1', mu=0, sd=SD, shape=(X.shape[1], n_hidden), testval=init_1)
    bias_1 = pm.Normal('bias1', mu=0, sd=SD, shape = n_hidden)
    weights_2 = pm.Normal('layer2', mu=0, sd=SD, shape=(n_hidden, n_hidden), testval=init_2)
    bias_2 = pm.Normal('bias2', mu=0, sd=SD, shape=n_hidden)
    weights_out = pm.Normal('out', mu=0, sd=SD, shape=(n_hidden, 1), testval=init_out)
    intercept = pm.Normal('intercept', mu=0, sd=SD)

    # Now assemble the neural network
    layer_1 = pm.math.tanh(pm.math.dot(ann_input, weights_1) + bias_1)
    layer_2 = pm.math.tanh(pm.math.dot(layer_1, weights_2) + bias_2)
    layer_out = pm.math.dot(layer_2, weights_out)

    y = pm.Normal('y', layer_out + intercept, observed=ann_output)

print "Done. Sampling..."

num_samples = 200
with neural_network:

    trace = pm.sample(num_samples, tune=num_samples, nuts_kwargs=dict(target_accept=0.999))

# for each of the num_samples parameter values sampled above, sample 500 times the expected y value.
samples = pm.sample_ppc(trace, model=neural_network, size=500)

y_preds = np.reshape(samples['y'], [num_samples, 500, X.shape[0]])

# get the average, since we're interested in plotting the expectation.
y_preds = np.mean(y_preds, axis=1)
y_preds = np.mean(y_preds, axis=0)

RMSD = np.sqrt(np.mean((y_preds - Y) ** 2.0))

plt.scatter(X, Y)
plt.scatter(X, y_preds, alpha=0.1)
plt.show()