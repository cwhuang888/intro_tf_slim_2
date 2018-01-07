import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class SpiralDataset(object):

    def __init__(self):
        self.class_num = None

    def generate_spiral_dataset(self, points_per_class=200, noise=0.2, random_seed=1):
        dataset = {}
        np.random.seed(random_seed)
        self.class_num = 3  # number of classes
        feature_dim = 2  # dimensionality of feature space
        dataset_size = points_per_class * self.class_num
        dataset['X'] = np.zeros((dataset_size, feature_dim), dtype='float32')
        dataset['y'] = np.zeros(dataset_size, dtype='uint8')

        for j in range(self.class_num):
            ix = range(points_per_class * j, points_per_class * (j + 1))
            r = np.linspace(0.0, 1, points_per_class)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, points_per_class) + \
                np.random.randn(points_per_class) * noise  # theta
            dataset['X'][ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            dataset['y'][ix] = j

        dataset['X'] = dataset['X'].astype(np.float64)
        dataset['y'] = dataset['y'].astype(np.int64)
        return dataset

    def plot_spiral(self, X_train, y_train, X_test):
        plt.title("Generated Spiral Dataset")
        plt.scatter(X_train[:, 0], X_train[:, 1],
                    c=y_train, s=30, cmap=plt.cm.rainbow)
        plt.scatter(X_test[:, 0], X_test[:, 1], s=30, alpha=0.8, c="black")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        # plot the resulting classifier
    def plot_classifier(self, model, dataset, checkpoint_dir, title):
        h = 0.02
        x_min, x_max = dataset['X'][:, 0].min() - 0.3, dataset['X'][:, 0].max() + 0.3
        y_min, y_max = dataset['X'][:, 1].min() - 0.3, dataset['X'][:, 1].max() + 0.3
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        with tf.Graph().as_default():
            inputs = tf.constant(np.c_[xx.ravel(), yy.ravel()])
            logits, end_points = model.graph(inputs)
            predictions = tf.argmax(logits, 1)

            # Make a session which restores the old parameters from a
            # checkpoint.
            sv = tf.train.Supervisor(logdir=checkpoint_dir)
            with sv.managed_session() as sess:
                inputs, predictions = sess.run([inputs, predictions])

        Z = predictions.reshape(xx.shape)
        plt.title(title)
        plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.8)
        plt.scatter(dataset['X'][:, 0], dataset['X'][:, 1], c=dataset['y'], s=40, cmap=plt.cm.rainbow)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()
