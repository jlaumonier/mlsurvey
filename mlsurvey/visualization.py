import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

"""Not tested because il will be transform to a external module (application ?)"""


class Visualization:

    def plot_data(self, x, y=None):
        fig, ax = plt.subplots()

        print(x)
        print(y)
        ax.scatter(x[:, 0], x[:, 1], c=y)
        ax.grid(True)
        fig.tight_layout()
        plt.draw()

    def plot_result(self, x, clf):
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        h = .05  # step size in the mesh

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        z = z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, z, cmap=cmap_light)
        plt.show()
