print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


def plot(X, Y=None):
    fig, ax = plt.subplots()

    print(X)
    print(Y)
    ax.scatter(X[:, 0], X[:, 1], c=Y)
    ax.grid(True)
    fig.tight_layout()
    plt.draw()


def plot_result(X, clf):
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    h = .05  # step size in the mesh

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.draw()


def nn():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Test = np.array([[0, 0]])
    plot(X)

    nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(Test)
    print(distances)

    nbrs2 = neighbors.NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(X)
    distances2, indices2 = nbrs2.kneighbors(Test)
    print(distances2)


def nnclassifier():
    iris = datasets.load_iris()

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    plot(X, y)

    n_neighbors = 15
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)

    plot_result(X, clf)


nn()
nnclassifier()
plt.show()
