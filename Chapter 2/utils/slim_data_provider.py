from utils.datasets import cifar10, flowers, text_datasets
import pandas as pd
import numpy as np
from collections import defaultdict
import tensorflow as tf
slim = tf.contrib.slim

class DatasetProvider(object):

    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    def preprocess_image(self, image, output_height, output_width):
        """Preprocesses the given image.
        Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.

        Returns:
        A preprocessed image.
        """
        tf.summary.image('image', tf.expand_dims(image, 0))
        # Transform the image to floats.
        image = tf.to_float(image)

        # Resize and crop if needed.
        resized_image = tf.image.resize_image_with_crop_or_pad(
            image, output_height, output_width)
        tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

        # Subtract off the mean and divide by the variance of the pixels.
        return tf.image.per_image_standardization(resized_image)

    def load_batch_cifar10(self, data_type, output_height, output_width, batch_size):
        dataset = cifar10.get_split(data_type, self.data_dir)
        return self.load_batch_CNN(dataset, data_type, output_height, output_width, batch_size)

    def load_batch_flowers(self, data_type, output_height, output_width, batch_size):
        dataset = flowers.get_split(data_type, self.data_dir)
        return self.load_batch_CNN(dataset, data_type, output_height, output_width, batch_size)

    def load_batch_CNN(self, dataset, data_type, output_height, output_width, batch_size=32):
        """Loads a single batch of data.

        Args:
          dataset: The dataset to load.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          batch_size: The number of images in the batch.

        Returns:
          images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
          labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
        """
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=8)
        image, label = data_provider.get(['image', 'label'])

        image = self.preprocess_image(image, output_height, output_width)

        # Batch it up.
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,  # one queue
            capacity=2 * batch_size)  # two batch per queue max

        return images, labels

    def load_batch_spiral(self, dataset):

        feature_dim = dataset['X'].shape[1]
        inputs = tf.constant(dataset['X'])
        inputs.set_shape([None, feature_dim])

        labels = tf.constant(dataset['y'])
        labels.set_shape([None])
        return inputs, labels

    def load_batch_dbpedia(self, batch_size, data_size):
        batch_indices = np.random.choice(data_size, batch_size, replace=False)
        return batch_indices

    def preprocess_vocabulary(self, x_train, x_test, max_document_length = 100):
        # Process vocabulary
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=2)
        x_train = np.array(list(vocab_processor.fit_transform(x_train)))
        x_test = np.array(list(vocab_processor.transform(x_test)))
        self.n_words = len(vocab_processor.vocabulary_)
        return x_train, x_test

    def get_dbpedia_dataset(self, size='small'):
        data = text_datasets.load_dbpedia(size=size)

        x_train = pd.DataFrame(data.train.data)[1]
        y_train = pd.Series(data.train.target)
        x_test = pd.DataFrame(data.test.data)[1]
        y_test = pd.Series(data.test.target)

        x_train, x_test = self.preprocess_vocabulary(x_train, x_test, max_document_length = 100)

        dataset = defaultdict(dict)
        dataset['train']['X'] = x_train
        dataset['train']['y'] = y_train
        dataset['test']['X'] = x_test
        dataset['test']['y'] = y_test
        return dataset
