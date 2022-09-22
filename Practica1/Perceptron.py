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
        self.w_ = np.zeros(1 + X.shape[1])
        for iter in range(self.n_iter):
            for i, x in enumerate(X):
                prediction = self.result(x)
                if prediction != y[i]:
                    self.errors_[iter] += 1
                self.update_weights(x, prediction, y[i])
        return self

    def update_weights(self, x, prediction, target):
        for w in range(x.shape[0]):
            deltaw = self.eta*(target - prediction)*x[w]
            self.w_[w+1] += deltaw
        self.w_[0] += self.eta*(target - prediction)

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """
        out = []
        for x in X:
            out.append(self.result(x))
        return out

    def result(self, x):
        z = np.dot(x, self.w_[1:])+self.w_[0]
        return 1 if z >=0 else -1
