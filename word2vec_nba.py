"""
Embedding for NBA news from 2017-2018 season
Author: Pau Bramon

This code was created following the Tensorflow tutorial and multiple online examples.
"""

import tensorflow as tf
import pau_utils
import pandas as pd
import pickle
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

OUTPUT_FOLDER = 'results/final_5'
DATAFILE = 'final_data2.csv'
USE_TEXT_COL = ['short_text', 'long_text']
VOCABULARY_SIZE = 15000
EMBEDDING_DIM = 200
WINDOW_SIZE_HALF = 1  # Default size of the windows. This is the number of target words on each side of the context word.
WINDOW_SPAN_HALF = 5  # Default span of the windows.
BATCH_SIZE = 128
NUM_NEGATIVE_EXAMPLES = 64
VALIDATION_SIZE = 16
VALIDATION_WINDOW = 100


def print_config(folder):
    configuration = """
        OUTPUT_FOLDER = {}
        DATAFILE = {}
        USE_TEXT_COL = {}
        VOCABULARY_SIZE = {}
        EMBEDDING_DIM = {}
        WINDOW_SIZE_HALF = {} 
        WINDOW_SPAN_HALF = {} 
        BATCH_SIZE = {}
        NUM_NEGATIVE_EXAMPLES = {}
        VALIDATION_SIZE = {}
        VALIDATION_WINDOW = {}
        """.format(OUTPUT_FOLDER, DATAFILE, str(USE_TEXT_COL), VOCABULARY_SIZE, EMBEDDING_DIM, WINDOW_SIZE_HALF,
                   WINDOW_SPAN_HALF, BATCH_SIZE, NUM_NEGATIVE_EXAMPLES, VALIDATION_SIZE, VALIDATION_WINDOW)
    with open(folder + "config.txt", "w") as text_file:
        text_file.write(configuration)


class word2vec(object):
    def __init__(self, data, vocabulary, reverse_vocabulary, embedding_dim=EMBEDDING_DIM, batch_size=BATCH_SIZE,
                 window_size_half=WINDOW_SIZE_HALF, window_span_half=WINDOW_SPAN_HALF,
                 num_negative_examples=NUM_NEGATIVE_EXAMPLES, valid_size=VALIDATION_SIZE,
                 valid_window=VALIDATION_WINDOW, output_folder=OUTPUT_FOLDER):
        self.x_ids = data
        self.vocabulary = vocabulary
        self.reverse_vocabulary = reverse_vocabulary
        self.vocabulary_size = max(vocabulary.word_index.values())
        self.embedding_dim = embedding_dim
        self.window_size_half = window_size_half
        self.window_span_half = window_span_half
        self.batch_size = batch_size
        self.num_negative_examples = num_negative_examples
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.output_folder = output_folder

        # initialize batch vectors
        self.batch_x = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        self.batch_y = np.ndarray(shape=(self.batch_size, self.window_size_half * 2), dtype=np.int32)

        # Initialize counters for the batch generation
        self.index_sent = 0
        self.index_word = 0
        self.epoch_counter = 0

    def create_network(self):
        # Create input variables
        self.input_x = tf.placeholder(tf.int32, [self.batch_size], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [self.batch_size, self.window_size_half * 2], name='input_y')

        # Create Embedding: [vocabulary_size, embedding_dim]
        with tf.name_scope('embedding'):
            init_width = 0.5 / self.embedding_dim
            self.embedding = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_dim], -1.0, 1.0), name="emb")
            self.embed = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # Construct the variables for the NCE loss
        with tf.name_scope('output_layer'):
            # Create output weights transposed
            self.out_w_t = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_dim],
                                                           stddev=1.0 / math.sqrt(self.embedding_dim)), name="out_w_t")
            # Create output bias
            self.out_b = tf.Variable(tf.zeros([self.vocabulary_size]), name="out_b")

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.out_w_t,
                    biases=self.out_b,
                    labels=self.input_y,
                    inputs=self.embed,
                    num_true=self.window_size_half * 2,
                    num_sampled=self.num_negative_examples,
                    num_classes=self.vocabulary_size))

        # Store values of loss for tensorboard
        tf.summary.scalar('loss', self.loss)

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        '''
        This code is from Thushan Ganegedara's implementation. Here we're going to choose a few common words and few 
        uncommon words. Then, we'll print out the closest words to them. It's a nice way to check that our embedding 
        table is grouping together words with similar semantic meanings. Source: https://github.com/vyomshm/Skip-gram-Word2vec
        '''
        # pick valid_size samples from (0,valid_window) and (1000-valid_window,1000) each ranges.  1000 is a randomly
        # picked value, since we now we have at least 1000 words in the vocabulary and these are not very frequent
        # (but not very rare either)
        not_usual = 1000
        self.valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size // 2))
        self.valid_examples = np.append(self.valid_examples,
                                        random.sample(range(not_usual - self.valid_window, not_usual),
                                                      self.valid_size // 2))

        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
        self.normalized_embedding = self.embedding / norm
        valid_embedding = tf.nn.embedding_lookup(self.normalized_embedding, valid_dataset)
        self.similarity = tf.matmul(valid_embedding, self.normalized_embedding, transpose_b=True)

    def generate_training_batch(self):
        '''
        This method generates the next training batch.
        :return: x_train and y_train. x_train will contain the context words to show at the input of the network.
        y_train will contain all the words to predict for each context word.
        '''

        creating_batch = True
        i = 0
        while creating_batch:
            if self.index_word - self.window_size_half >= 0:
                if self.index_word + self.window_size_half < len(self.x_ids[self.index_sent]):
                    self.batch_x[i] = self.x_ids[self.index_sent][self.index_word]
                    span_l = self.x_ids[self.index_sent][
                             max(0, self.index_word - self.window_span_half):self.index_word]
                    self.batch_y[i, 0:self.window_size_half] = random.sample(span_l,self.window_size_half)
                    span_r = self.x_ids[self.index_sent][self.index_word + 1:min(len(self.x_ids[self.index_sent]),
                                                                                 self.index_word + self.window_span_half + 1)]
                    self.batch_y[i, self.window_size_half:] = random.sample(span_r,self.window_size_half)
                    i += 1
                    self.index_word += 1
                else:
                    self.index_word = 0
                    self.index_sent += 1
                    if self.index_sent >= len(self.x_ids):
                        self.index_sent = 0
                        self.epoch_counter += 1
            else:
                self.index_word += 1

            if i >= self.batch_x.shape[0]:
                creating_batch = False
        self.batch_x = self.batch_x.reshape(-1, )

    def plot_with_labels(self, low_dim_embs, labels, filename):
        '''
        Function to draw visualization of distance between embeddings. Source: Tensorflow tutorial
        :param labels: labels of the embedding
        :param filename: name of the file to save the figure
        :return:
        '''
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

        plt.savefig(filename)

    def train(self):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            with sess.as_default():
                # Create network
                self.create_network()

                # Define training
                with tf.name_scope('train'):
                    # optimizer = tf.train.GradientDescentOptimizer(1.0)
                    optimizer = tf.train.AdamOptimizer()
                    self.train_step = optimizer.minimize(self.loss)

                # Initialize variables
                sess.run(tf.global_variables_initializer())

                # Prepare writer
                summaries = tf.summary.merge_all()

                # Create a saver.
                saver = tf.train.Saver()

                writer = tf.summary.FileWriter(self.output_folder, sess.graph)
                writer.add_graph(sess.graph)

                # Step 5: Begin training.
                num_steps = 500000

                average_loss = 0
                for step in xrange(num_steps):
                    self.generate_training_batch()

                    # Define metadata variable.
                    run_metadata = tf.RunMetadata()

                    feed_dict = {self.input_x: self.batch_x, self.input_y: self.batch_y}

                    _, summary, loss_val = sess.run(
                        [self.train_step, summaries, self.loss],
                        feed_dict=feed_dict,
                        run_metadata=run_metadata)
                    average_loss += loss_val

                    # Add returned summaries to writer in each step.
                    writer.add_summary(summary, step)
                    # Add metadata to visualize the graph for the last run.
                    if step == (num_steps - 1):
                        writer.add_run_metadata(run_metadata, 'step%d' % step)

                    if step % 2000 == 0:
                        if step > 0:
                            average_loss /= 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        output_log = "** Average loss {:g}, step {:g}, epoch {:g}".format(average_loss, step,
                                                                                          self.epoch_counter)
                        print(output_log)
                        with open(self.output_folder + "/output_log.txt", "a") as myfile:
                            myfile.write(output_log + '\n')
                        average_loss = 0

                    # Note that this is expensive
                    if step % 10000 == 0:
                        sim = self.similarity.eval()
                        for i in xrange(self.valid_size):
                            valid_word = self.reverse_vocabulary[self.valid_examples[i]]
                            top_k = 8  # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = 'Nearest to %s:' % valid_word
                            for k in xrange(top_k):
                                close_word = self.reverse_vocabulary[nearest[k]]
                                log_str = '%s %s,' % (log_str, close_word)
                            print(log_str)
                            with open(self.output_folder + "/output_log.txt", "a") as myfile:
                                myfile.write(log_str + '\n')

                self.final_embeddings = self.normalized_embedding.eval()

                # Write corresponding labels for the embeddings.
                with open(self.output_folder + '/metadata.tsv', 'w') as f:
                    for i in xrange(self.vocabulary_size):
                        if i == 0:
                            f.write('Labels' + '\n')
                        else:
                            f.write(self.reverse_vocabulary[i] + '\n')

                # Save the model for checkpoints.
                saver.save(sess, os.path.join(self.output_folder, 'model.ckpt'))

                # Create a configuration for visualizing embeddings with the labels in TensorBoard.
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = self.embedding.name
                embedding_conf.metadata_path = os.path.join('metadata.tsv')
                projector.visualize_embeddings(writer, config)
            writer.close()

        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_only, :])
        labels = [self.reverse_vocabulary[i + 1] for i in xrange(plot_only)]
        self.plot_with_labels(low_dim_embs, labels, os.path.join(self.output_folder, 'tsne.png'))


def prepare_data(datafile=DATAFILE, use_text_col=USE_TEXT_COL, output_path=OUTPUT_FOLDER,
                 vocabulary_size=VOCABULARY_SIZE, store_vocabulary=True, subsample_most_frequent=True):
    # Gather data
    data = pd.read_csv(datafile)
    sentences = []
    for col_text in use_text_col:
        sentences = sentences + list(data[col_text])

    # Generate vocabulary
    vocabulary = Tokenizer(num_words=vocabulary_size, filters='!"%&()*,./:;?@[\\]^_{|}~\t\n')
    vocabulary.fit_on_texts(sentences)
    x_ids = vocabulary.texts_to_sequences(sentences)

    # Generate reverse vocabulary
    reverse_vocabulary = {v: k for k, v in vocabulary.word_index.iteritems()}
    # Add unknown word
    reverse_vocabulary[0] = 'unknown'

    if subsample_most_frequent:
        # Set a threshold such that the minimum probability is 0.5
        max_count = max(vocabulary.word_counts.values())
        min_prob = 0.5
        threshold = min_prob * min_prob / max_count

        # Get probabilities for all words
        p_drop = {}
        for key, val in vocabulary.word_index.iteritems():
            p_drop[val] = 1 - math.sqrt(threshold * vocabulary.word_counts[key])

        # Subsample words
        x_ids_sub = []
        for i in xrange(len(x_ids)):
            x_ids_sub.append([word for word in x_ids[i] if p_drop[word] > np.random.random()])
        x_ids = x_ids_sub
        del x_ids_sub

    # Save vocabulary
    if store_vocabulary:
        with open(output_path + '/' + 'initial_data', "wb") as f:
            pickle.dump((x_ids, vocabulary.word_index, reverse_vocabulary), f, protocol=2)

    return x_ids, vocabulary, reverse_vocabulary


def main():
    # Create folder if it doesn't exist
    pau_utils.create_folder(OUTPUT_FOLDER + '/')

    print_config(OUTPUT_FOLDER + '/')

    x_ids, vocabulary, reverse_vocabulary = prepare_data()
    w2v = word2vec(x_ids, vocabulary, reverse_vocabulary)
    w2v.train()
    return w2v


if __name__ == "__main__":
    w2v = main()
