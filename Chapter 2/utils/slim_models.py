from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


class Model(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.graph = None

    def examine_model_structure(self):
        with tf.Graph().as_default():

            inputs = tf.placeholder(tf.float32, shape=(None,) + self.input_dim)

            # Build model
            _, end_points = self.graph(inputs)

            # Print name and shape of each tensor.
            print("Layers")
            for k, v in end_points.iteritems():
                print('name = {} {}shape = {}'.format(
                    v.name, " " * (55 - len(v.name)), v.get_shape()))

            # Print name and shape of parameter nodes
            print("\n")
            print("Parameters")
            for v in slim.get_model_variables():
                print('name = {} {}shape = {}'.format(
                    v.name, " " * (55 - len(v.name)), v.get_shape()))


class DeepClassifier(Model):

    def __init__(self, input_dim, output_dim):
        Model.__init__(self, input_dim, output_dim)
        self.graph = self.deep_classifier

    def deep_classifier(self, inputs, weight_decay=0.005, dropout=None, is_training=None, scope="deep_classifier"):

        with tf.variable_scope(scope, 'deep_classifier', [inputs]) as vs:
            end_points_collection = vs.original_name_scope + '_end_points'

            # Set the default weight _regularizer and acvitation for each
            # fully_connected layer.
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(
                                    weight_decay),
                                outputs_collections=end_points_collection):

                # Creates a fully connected layer from the inputs with 64
                # hidden units.
                net = slim.fully_connected(inputs, 64, scope='fc1')

                # Adds another fully connected layer with 32 hidden units.
                net = slim.fully_connected(net, 32, scope='fc2')

                # Creates a fully-connected layer with a single hidden unit. Note that the
                # layer is made linear by setting activation_fn=None.
                # override the arg_scope variables
                net = slim.fully_connected(
                    net, self.output_dim, activation_fn=None, scope='prediction')
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return net, end_points


class CNNClassifier(Model):

    def __init__(self, dataset_name, input_dim, output_dim):
        Model.__init__(self, input_dim, output_dim)
        if dataset_name == "flowers":
            self.graph = self.CNN_flowers_classifier
        elif dataset_name == "cifar10":
            self.graph = self.CNN_cifar10_classifier
        elif dataset_name == "dbpedia":
            self.graph = self.CNN_dbpedia_text_classifier
        else:
            raise ValueError('dataset_name [%s] was not recognized.')

    def CNN_arg_scope(self, weight_decay):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=0.01)):

            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def CNN_dbpedia_text_classifier(self, inputs, weight_decay=0.0005, is_training=True,
                                    dropout=0.5, spatial_squeeze=True, scope='CNN_dbpedia_text_classifier'):
        with slim.arg_scope(self.CNN_arg_scope(weight_decay)):
            with tf.variable_scope(scope, 'CNN_dbpedia_text_classifier', [inputs]) as vs:
                end_points_collection = vs.original_name_scope + '_end_points'

                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    branches = []
                    for i, filter_size in enumerate([3, 4, 5]):
                        with tf.name_scope("conv-maxpool-%s" % filter_size):
                            embedding_size = 20
                            num_filters = 10
                            seq_length = 100
                            net = slim.conv2d(inputs, num_filters, [filter_size, embedding_size], stride=[1, 1],
                                              padding="VALID", scope='1D-conv_%d' % (i + 1))
                            net = slim.max_pool2d(
                                net, [seq_length - filter_size + 1, 1], stride=[1, 1], scope="1D-pool-%d" % (i + 1))
                            branches.append(net)
                    net = tf.concat(branches, 3)
                    net = slim.dropout(
                        net, 0.5, is_training=is_training, scope='dropout4')

                    net = slim.conv2d(net, self.output_dim, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='prediction')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)
                    if spatial_squeeze:
                        net = tf.squeeze(
                            net, [1, 2], name='prediction/squeezed')
                        end_points[vs.name + '/prediction'] = net
                    return net, end_points

    def CNN_cifar10_classifier(self, inputs, weight_decay=0.0005, is_training=True,
                               dropout=0.5, spatial_squeeze=True, scope='CNN_cifar10_classifier'):
        """
        CNN_classifier Example.
        """

        with slim.arg_scope(self.CNN_arg_scope(weight_decay)):
            with tf.variable_scope(scope, 'CNN_cifar10_classifier', [inputs]) as vs:
                end_points_collection = vs.original_name_scope + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(inputs, 2, slim.conv2d,
                                      64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128,
                                      [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')

                    # Use conv2d instead of fully_connected layers.
                    net = slim.conv2d(
                        net, 256, [8, 8], padding='VALID', scope='fc3')
                    net = slim.dropout(net, dropout,
                                       is_training=is_training, scope='dropout3')

                    net = slim.conv2d(net, self.output_dim, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='prediction')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)
                    if spatial_squeeze:
                        net = tf.squeeze(
                            net, [1, 2], name='prediction/squeezed')
                        end_points[vs.name + '/prediction'] = net
                    return net, end_points

    def CNN_flowers_classifier(self, inputs, is_training=True, weight_decay=0.0005,
                               dropout=0.5, spatial_squeeze=True, scope='CNN_flowers_classifier'):
        """
        CNN_classifier Example.
        """
        with slim.arg_scope(self.CNN_arg_scope(weight_decay)):
            with tf.variable_scope(scope, 'CNN_flowers_classifier', [inputs]) as vs:
                end_points_collection = vs.original_name_scope + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    net = slim.repeat(inputs, 1, slim.conv2d, 64,
                                      [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')

                    net = slim.repeat(net, 2, slim.conv2d, 128,
                                      [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')

                    net = slim.repeat(net, 3, slim.conv2d, 256,
                                      [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    # Use conv2d instead of fully_connected layers.
                    net = slim.conv2d(
                        net, 1024, [8, 8], padding='VALID', scope='fc4')
                    net = slim.dropout(net, dropout, is_training=is_training,
                                       scope='dropout4')

                    net = slim.conv2d(net, 1024, [1, 1], scope='fc5')
                    net = slim.dropout(net, dropout, is_training=is_training,
                                       scope='dropout5')

                    net = slim.conv2d(net, self.output_dim, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='prediction')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)
                    if spatial_squeeze:
                        net = tf.squeeze(
                            net, [1, 2], name='prediction/squeezed')
                        end_points[vs.name + '/prediction'] = net
                    return net, end_points
