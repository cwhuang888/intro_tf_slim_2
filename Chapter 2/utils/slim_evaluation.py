import tensorflow as tf
slim = tf.contrib.slim


class ModelEvaluater(object):

    def __init__(self, model, dataset_provider, data_name, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.dataset_provider = dataset_provider
        self.data_name = data_name
