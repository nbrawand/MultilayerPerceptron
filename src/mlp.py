import numpy as np
import scipy
from scipy.special import expit
import sys

class network:
    """Implements perceptron network with 1 hidden layer.

    Variables
    ---------
    weights - [w1,w2]
    w1 - [nHidden, nFeatures+1] +1 from the bias unit
    w2 - [nOutputs, nHidden+1] +1 from the bias unit
    nHidden - number of nodes in hidden layer
    nFeatures - number of features in input
    nOutputs - number of output units should be == number of classes of labels

    """

    def __init__(self, nFeatures, nOutputs, nHidden=30):

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nOutputs = nOutputs

        self.InitializeWeights()

    def InitializeWeights(self):
        """Initialize the weight matrix."""

        self.weights = [[],[]]

        # create first set of weights w1
        self.weights[0] = np.random.uniform(-1, 1, (self.nFeatures+1)*self.nHidden)
        self.weights[0] = self.weights[0].reshape((self.nHidden, (self.nFeatures+1)))

        # create 2nd set of weights w2
        self.weights[1] = np.random.uniform(-1, 1, self.nOutputs*(self.nHidden+1))
        self.weights[1] = self.weights[1].reshape((self.nOutputs, (self.nHidden+1)))

        return self.weights

    def train(self, X, labs, epochs, eta):
        """Train the model. Loop over epochs: Feedforward, backprop.

        args
        ----
        X - training set [Samples, self.nFeatures]
        labs - labels for X [Samples]
        epochs - number of epochs for learning
        eta - learning rate

        return
        ------
        self

        """
        onehot = self.InitializeOnehot(labs)

        for epoch in range(epochs):

            # print progress
            sys.stderr.write('\repoch: {}/{}'.format(epoch+1, epochs))
            sys.stderr.flush()

            a1, z2, a2, z3, a3 = self.FeedForward(X)
            self.UpdateWeights(a1, z2, a2, z3, a3, onehot, eta)

        return self

    def AddBiasUnit(self, X, to2ndIndex=True):
        """Adds the bias unit to 2d matrix X. If to2ndIndex=True, then add to 2nd dim else 1st."""
        if to2ndIndex:
            newX = np.ones((X.shape[0], X.shape[1] + 1))
            newX[:, 1:] = X
        else:
            newX = np.ones((X.shape[0] + 1, X.shape[1]))
            newX[1:, :] = X
        return newX

    def FeedForward(self, X):
        """feed input through network.

        args
        ----
        X - input to feed through network [nSamples, nFeatures]

        return
        ------
        a1 - X+bias unit activation of input layer
        z2 - input for hidden layer
        a2 - activation of hidden layer
        z3 - input for output layer
        a3 - activation of output layer

        """
        #
        # do first layer
        #
        # add bias to samples X+bias = a1
        # a1 - [nSamples, (nFeatures+1)]
        a1 = self.AddBiasUnit(X)

        # net input for hidden layer
        # z2=W1.a1
        # z2 - [nHidden, nSamples]
        z2 = self.weights[0].dot(a1.transpose())

        # activation of hidden layer
        # a2 = phi(z2)
        # a2 - [nHidden, nSamples]
        a2 = self.Activation(z2)

        # add bias to activation of hidden layer
        # a2 - [(nHidden+1), nSamples]
        a2 = self.AddBiasUnit(a2, to2ndIndex=False)

        # net input for output layer
        # z3 = W2.a2
        # z3 - [nOutputs, nSamples]
        z3 = self.weights[1].dot(a2)

        # activation  of output layer
        # a3 = phi(z3)
        # a3 - [nOutputs, nSamples]
        a3 = self.Activation(z3)

        return a1, z2, a2, z3, a3


    def UpdateWeights(self, a1, z2, a2, z3, a3, onehot, eta):
        """Updates self.weights. Do backprop and update self.weights.

        args
        ----
        a1 - X+bias unit activation of input layer
        z2 - input for hidden layer
        a2 - activation of hidden layer
        z3 - input for output layer
        a3 - activation of output layer

        """

        # error of output vector
        # d3 = a3 - y (where y is the onehot rep of the true labels)
        # d3 - [nOutputs, nSamples]
        d3 = a3 - onehot

        # grad2 - [nOutputs, nHidden+1]
        grad2 = d3.dot(a2.T)



        # error in the hidden layer
        # d2 = w2^T . d3 * (d phi(z2)/d z2)

        # add bias unit to z2
        # z2 - [hidden+1, samples]
        z2 = self.AddBiasUnit(z2, to2ndIndex=False)

        phiZ2Grad = self.ActivationGrad(z2)

        # W2^T . d3 * grad(phi(z2))
        # ([nOutputs, nHidden+1]^T . [nOuputs, nSamples])
        #                             * [nHidden+1,  nSamples]
        d2 = self.weights[1].transpose().dot(d3) * phiZ2Grad

        # remove hidden layer
        d2 = d2[1:,:]

        # grad1 - [nHidden, features+1]
        grad1 = d2.dot(a1)

        deltas = [ grad*eta for grad in [grad1, grad2] ]

        self.weights = [weight - delta for weight, delta in zip(self.weights, deltas)]


    def Activation(self, Z):
        """Apply activation function to input: expit(x) = 1/(1+exp(-x))"""
        return expit(Z)


    def ActivationGrad(self, Z):
        """Calculate grad of Activation"""
        tmp = self.Activation(Z)
        return tmp * (1.0 - tmp)


    def InitializeOnehot(self, labs):
        """Initialize one hot rep for labels.

        args
        ----
        labs - labels [nSamples]

        return
        ------
        matrix one-hot rep of labels [nOutput, NSamples]

        """

        onehot = np.zeros((self.nOutputs, len(labs)))

        for i, val in enumerate(labs):
            onehot[val, i] = 1.0

        return onehot

    def predict(self, X):
        """Predict labels. Feedforward X.

        args
        ----
        X - [nSamples, nFeatures]

        return
        ------
        pred - label predictions [n_samples, nOutputs]

        """
        a1, z2, a2, z3, a3 = self.FeedForward(X)
        pred = np.argmax(z3, axis=0)
        return pred
