r"""Downloads and converts a particular dataset.

```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils.datasets import download_and_convert_cifar10
from utils.datasets import download_and_convert_flowers
from utils.datasets import download_and_convert_mnist

class DownloaderAndConverter(object):

    def download_and_convert(self, dataset_name, dataset_dir):

      if dataset_name == 'cifar10':
        download_and_convert_cifar10.run(dataset_dir)
      elif dataset_name == 'flowers':
        download_and_convert_flowers.run(dataset_dir)
      elif dataset_name == 'mnist':
        download_and_convert_mnist.run(dataset_dir)
      else:
        raise ValueError(
            'dataset_name [%s] was not recognized.' % dataset_dir)
