import os
import time
import tensorflow as tf
import pickle

from model import Model
from q2_initialization import xavier_weight_init
from general_utils import Progbar
from parser_utils import minibatches, load_and_preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # 这两个值大小取决于parser的config的配置
    n_features = 36  # use_pos=True && use_dep=False 为 36
    # use_pos=True && use_dep=True  为 48
    # use_pos=False && use_dep=False  为 18

    n_classes = 3  # unlabeled=True 时为 3，
    # debug=False && unlabeled=False 为 79;
    # debug=True && unlabeled=False 为 77；

    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 100
    lr = 0.001


class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.n_features])
        self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.n_classes])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.beta_regul = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1, beta_regul=10e-7):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.input_placeholder: inputs_batch, \
                     self.dropout_placeholder: dropout, \
                     self.beta_regul: beta_regul
                     }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        embed_matrix = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embed_matrix, self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.config.n_features * self.config.embed_size])
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features*embed_size, hidden_size)
                    b1: (hidden_size,)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)
        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument. 
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        x = self.add_embedding()
        xavier = xavier_weight_init()
        with tf.variable_scope("transformation"):
            self.W = W = xavier([self.config.n_features * self.config.embed_size, self.config.hidden_size])
            b1 = tf.Variable(tf.random_uniform([self.config.hidden_size]))
            z1 = tf.matmul(x, W) + b1
            h = tf.nn.relu(z1)
            h_drop = tf.nn.dropout(h, self.config.dropout)

            U = xavier([self.config.hidden_size, self.config.n_classes])
            b2 = tf.Variable(tf.random_uniform([self.config.n_classes]))
            pred = tf.matmul(h_drop, U) + b2

        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder)
        # loss += self.beta_regul*tf.nn.l2_loss(self.W)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    # 对训练集进行训练，得到的模型预测开发集
    def run_epoch(self, sess, parser, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])
        print("Evaluating on dev set",)
        dev_score, _ = parser.parse(dev_set)
        print("- dev_score: {:.2f}".format(dev_score * 100.0))
        return dev_score

    def fit(self, sess, saver, parser, train_examples, dev_set):
        best_dev_score = 0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_score = self.run_epoch(sess, parser, train_examples, dev_set)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if saver:
                    print("New best dev score! Saving model in ./data/weights/parser.weights")
                    saver.save(sess, './data/weights/parser.weights')
            print

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=True):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            print("Building model...", )
            start = time.time()
            model = ParserModel(config, embeddings)
            parser.model = model
            print("took {:.2f} seconds\n".format(time.time() - start))

            init = tf.global_variables_initializer()
            # If you are using an old version of TensorFlow, you may have to use
            # this initializer instead.
            # init = tf.initialize_all_variables()
            saver = None if debug else tf.train.Saver()

            with tf.Session() as session:
                parser.session = session
                session.run(init)

                print(80 * "=")
                print("TRAINING")
                print(80 * "=")
                model.fit(session, saver, parser, train_examples, dev_set)

                if not debug:
                    print(80 * "=")
                    print("TESTING")
                    print(80 * "=")
                    print("Restoring the best model weights found on the dev set")
                    saver.restore(session, './data/weights/parser.weights')
                    print("Final evaluation on test set", )
                    score, dependencies = parser.parse(test_set)
                    print("- test score: {:.2f}".format(score * 100.0))
                    print("Writing predictions")
                    with open('q2_test.predicted.pkl', 'wb') as f:
                        pickle.dump(dependencies, f, -1)
                    print("Done!")

if __name__ == '__main__':
    main(False)
    with open('q2_test.predicted.pkl', 'rb') as f:
        dep = pickle.load(f)
    pass
