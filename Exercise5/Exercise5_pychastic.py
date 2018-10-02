import numpy as np
import matplotlib.pyplot as plt
from Pychastic import BayesianNeuralNetwork as BNN

# 1. generate a simple non-linear function
X = np.reshape(np.arange(-5.0, 5.0, 0.01), [-1, 1])

print "shape of X: ", X.shape

Y = X ** 2.0

plt.scatter(X, Y)
plt.show()

# 2. neural network
print "Building neural network..."

nn = BNN.BayesianNeuralNetwork([5, 5], inference_method='mcmc')
nn.fit(X, Y, samples=200)
y_preds = nn.predict(X)
RMSD = nn.RMSD(X, Y)

print "Root Mean Square deviation:", RMSD

plt.scatter(X, Y)
plt.scatter(X, y_preds, alpha=0.1)
plt.show()
