import numpy as np


class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.errors_ = np.zeros(self.n_iter)

    def fit(self, X, y):

        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        # Initiate weights: one weight for each dimension plus an extra one
        # for the threshold
        self.w_ = np.zeros(1 + X.shape[1])

        # For each epoch, we do:
        for iter in range(self.n_iter):
            # For each entry:
            for i, x in enumerate(X):
                # We get the prediction of the entry
                prediction = self.compute(x)
                # If the prediction is incorrect, we increase the number of errors in this epoch
                if prediction != y[i]:
                    self.errors_[iter] += 1
                # We update the weights with the entry, prediction and expected value
                self.update_weights(x, prediction, y[i])
        return self

    def update_weights(self, x, prediction, target):
        # For each dimension of the entry, we update the weight
        for w in range(x.shape[0]):
            deltaw = self.eta * (target - prediction) * x[w]
            self.w_[w + 1] += deltaw
        # Finally we update the weight of the threshold
        self.w_[0] += self.eta * (target - prediction)

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """
        out = []
        # For each entry in the set, we compute the value and store it in the output list
        for x in X:
            out.append(self.compute(x))
        return out

    def activate(self, z):
        # Step function
        return 1 if z >= 0 else -1

    def compute(self, x):
        # Computes weights * entry + threshold weight, and returns
        z = np.dot(x, self.w_[1:]) + self.w_[0]
        return self.activate(z)
