import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.spatial import distance_matrix
from sklearn.preprocessing import PolynomialFeatures

def kernel_lineal(x1, x2):
    return x1.dot(x2.T)


def kernel_gaussian(x1, x2, gamma=1):
    return np.exp(-gamma * distance_matrix(x1, x2)**2)


def kernel_polynomial(x1, x2, d=3, r=0):
    return (x1.dot(x2.T) + r)**d


# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=3, n_clusters_per_class=1, class_sep=2,
                           random_state=9)
y[y == 0] = -1

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

#Entrenam una SVM linear (classe SVC)
# TODO
svm = SVC(kernel="linear", random_state=2,C=1)
svm.fit(X_transformed, y)
#Entrenam una SVM linear amb la nostra funcio kernel_lineal

svm2 = SVC(kernel=kernel_lineal, random_state=2,C=1)
svm2.fit(X_transformed, y)

predictions = svm.predict(X_transformed)
predictions2 = svm2.predict(X_transformed)

accuracy = np.mean(predictions == y)
accuracy2 = np.mean(predictions2 == y)

print("Accuracy SVM linear: ", accuracy)
print("Accuracy SVM linear amb kernel personalitzat: ", accuracy2)
print("")

np.random.seed()
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_xor_transformed = scaler.fit_transform(X_xor)

svmGauss = SVC(kernel="rbf",C=1)
svmGauss.fit(X_xor_transformed, y_xor)

svmGauss2 = SVC(kernel=kernel_gaussian,C=1)
svmGauss2.fit(X_xor_transformed, y_xor)

predictionsGauss = svmGauss.predict(X_xor_transformed)
predictionsGauss2 = svmGauss2.predict(X_xor_transformed)

accuracyGauss = np.mean(predictionsGauss == y_xor)
accuracyGauss2 = np.mean(predictionsGauss2 == y_xor)

print("Accuracy SVM gauss: ", accuracyGauss)
print("Accuracy SVM gauss amb kernel personalitzat: ", accuracyGauss2)
print("")
svmPoly = SVC(kernel="poly", C=10, degree=3, gamma=1)
svmPoly.fit(X_transformed, y)

svmPoly2 = SVC(kernel=kernel_polynomial, C=10, gamma=1)
svmPoly2.fit(X_transformed, y)

predictionsPoly = svmPoly.predict(X_transformed)
predictionsPoly2 = svmPoly2.predict(X_transformed)

accuracyPoly = np.mean(predictionsPoly == y)
accuracyPoly2 = np.mean(predictionsPoly2 == y)

print("Accuracy SVM poly: ", accuracyPoly)
print("Accuracy SVM poly amb kernel personalitzat: ", accuracyPoly2)

# Use PolynomialFeatures
X_newfeatures = PolynomialFeatures(degree=3).fit_transform(X_transformed)
svmLinealPoly = SVC(kernel="linear", C=1)
svmLinealPoly.fit(X_newfeatures, y)

predictionsLinealPoly = svmLinealPoly.predict(X_newfeatures)

accuracyLinealPoly = np.mean(predictionsLinealPoly == y)

print("Accuracy SVM lineal amb features polinomials: ", accuracyLinealPoly)


