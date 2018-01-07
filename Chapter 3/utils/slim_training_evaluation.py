import tensorflow as tf
slim = tf.contrib.slim


class ModelTrainerEvaluater(object):

    def __init__(self, model, dataset_provider, data_name, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.dataset_provider = dataset_provider
        self.data_name = data_name

    def train_classifier(self, inputs, targets, weight_decay, dropout=0.5, learning_rate=0.005, iterations=10, log_frq=5):
        # Make the model.
        logits, nodes = self.model.graph(
            inputs, weight_decay=weight_decay, dropout=dropout, is_training=True)

        # Add the loss function to the graph.
        one_hot_labels = slim.one_hot_encoding(targets, self.model.output_dim)
        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)

        # The total loss is the uers's loss plus any regularization losses.
        total_loss = slim.losses.get_total_loss()
        # total_loss = tf.losses.get_losses()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Run the training inside a session.
        final_loss = slim.learning.train(
            train_op,
            logdir=self.checkpoint_dir,
            number_of_steps=iterations,
            save_summaries_secs=5,
            log_every_n_steps=log_frq)

        print("Finished training. Last batch loss:", final_loss)
        print("Checkpoint saved in %s" % self.checkpoint_dir)

    def get_data(self, dataset, data_type, batch_size):
        if self.data_name == "spiral":
            inputs, targets = self.dataset_provider.load_batch_spiral(dataset)
        elif self.data_name == "cifar10":
            height, width, _ = self.model.input_dim
            inputs, targets = self.dataset_provider.load_batch_cifar10(
                data_type, height, width, batch_size)
        elif self.data_name == "flowers":
            height, width, _ = self.model.input_dim
            inputs, targets = self.dataset_provider.load_batch_flowers(
                data_type, height, width, batch_size)
        else:
            raise ValueError(
                "data_name: [%s] isn't 'spiral', 'flowers', 'cifar10', or 'mnist'" % self.data_name)
        return inputs, targets

    def train(self, weight_decay, learning_rate, iterations, log_frq, dropout=None,
              dataset=None, data_type='train', batch_size=None):

        self.dropout = dropout

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            inputs, targets = self.get_data(dataset, data_type, batch_size)

            self.train_classifier(
                inputs, targets, weight_decay, dropout, learning_rate, iterations, log_frq)

    def evaluate_classifier(self, inputs, targets, num_evals):

        logits, nodes = self.model.graph(
            inputs, dropout=self.dropout, is_training=False)
        predictions = tf.argmax(logits, 1)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, targets),
            'eval/Recall@3': slim.metrics.streaming_sparse_recall_at_k(tf.to_float(logits), tf.expand_dims(targets, 1), 3),
            'eval/Precision': slim.metrics.streaming_precision(predictions, targets),
            'eval/Recall': slim.metrics.streaming_recall(predictions, targets)
        })

        print('Running evaluation Loop...')
        checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        metric_values = slim.evaluation.evaluate_once(
            num_evals=num_evals,
            master='',
            checkpoint_path=checkpoint_path,
            logdir=self.checkpoint_dir,
            eval_op=names_to_updates.values(),
            final_op=names_to_values.values())

        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))

    def evaluate(self, num_evals, dataset=None, data_type='validation', dropout=None, batch_size=None):

        self.dropout = dropout

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            inputs, targets = self.get_data(dataset, data_type, batch_size)

            self.evaluate_classifier(inputs, targets, num_evals)
