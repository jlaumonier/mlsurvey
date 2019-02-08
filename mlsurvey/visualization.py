import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Visualization:

    @staticmethod
    def plot_data(x, y=None):
        """
        Create a matplotlib figure from
        :param x: 2D array for x values
        :param y: 1D array for the class
        """
        cm_points = ListedColormap(['#FF0000', '#FFFFFF', '#00FF00', '#000000', '#0000FF'])
        ax = plt.subplot(1, 2, 2)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_points, edgecolors='k')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.show()

    @staticmethod
    def creation_mesh(x):
        """
        Create a grid on bidimensionnal space. Calculate grid from min and max of each dimension with 0.02 resolution
        :param x: two columns vectors
        """
        h = .02  # step size in the mesh
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    @staticmethod
    def plot_result(x, y, clf, name, multi=False):
        """
        Display classifier result on a dataset
        :param x: 2D array for x values
        :param y: 1D array for the class
        :param clf: classifier
        :param name: name of the graph
        :param multi: display decision frontier
        """

        cm = 'jet_r'
        # regions and bounds calculus
        xx, yy = Visualization.creation_mesh(x)
        # multiclass (2+)
        if multi:
            z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            if hasattr(clf, "decision_function"):
                z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        z = z.reshape(xx.shape)
        ax = plt.subplot(1, 2, 1)
        test = ax.contourf(xx, yy, z, 100, cmap=cm, alpha=.8)

        cbar = plt.colorbar(test)
        cbar.ax.set_title('score')
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm,
                   edgecolors='k', s=100)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(name, fontsize=22)

        plt.draw()
