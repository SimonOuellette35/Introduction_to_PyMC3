import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from Pychastic import StochasticNeuralNetwork as SNN

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

nn = SNN.StochasticNeuralNetwork([3, 3], 20)
nn.fit(X, Y, samples=1000, advi_n=20000)
y_preds = nn.predict(X)
RMSD = nn.RMSD(X, Y)

print "Root Mean Square deviation:", RMSD

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