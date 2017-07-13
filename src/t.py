import cleanmlp
import numpy as np


nsam = 4
xsize = 10
outputs = 6

X = np.random.uniform(0, 1, nsam*xsize).reshape((nsam, xsize))

nn = cleanmlp.network(xsize, outputs, nHidden=2)

y = np.random.random_integers(0, outputs-1, nsam)


nn.train(X, y, 5, 0.001)
