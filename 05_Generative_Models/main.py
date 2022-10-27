import sys

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Function to get the number of components needed to explain 95% of the variance
def get_n_components(var_ratio, min_variance, delta_between_components):
    for i in range(len(var_ratio)):
        if var_ratio[i] >= min_variance:
            if var_ratio[i] - var_ratio[i-1] < delta_between_components:
                return i
    return len(var_ratio)


def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


# Load the data
digits = datasets.load_digits()
plot_digits(digits.data[:100, :])

#Do the PCA analysis to see how many components are needed
#scaler = StandardScaler()
X_scaled = digits.data #scaler.fit_transform(digits.data)
pca = PCA()
pca.fit(X_scaled)

# Plot the explained variance
fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('number of components')
ax.set_ylabel('cumulative explained variance')
n_components = get_n_components(np.cumsum(pca.explained_variance_ratio_), 0.95, 0.025)
plt.axvline(n_components, c='red')
print(f'Optimal number of dimensions: {n_components}')
plt.show()

# Do the PCA analysis with the number of components needed
pca = PCA(n_components=n_components)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

# Choose number of gaussians
min = sys.maxsize
n_gaussians = 0
bics = []
for i in range(1, 10):
    gmm = GaussianMixture(n_components=i, random_state=42)
    gmm.fit(X_pca)
    bics.append(gmm.bic(X_pca))
    if gmm.bic(X_pca) < min:
        min = gmm.bic(X_pca)
        n_gaussians = i

plt.plot(range(1, 10), bics)
plt.show()
print(f'Optimal number of gaussians: {n_gaussians}')

GMM = GaussianMixture(n_components=n_gaussians)
gmm.fit(X_pca)
new_digits = pca.inverse_transform(gmm.sample(100)[0])
plot_digits(new_digits)
plt.show()

