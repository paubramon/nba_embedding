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
import numpy as np
import random

OUTPUT_FOLDER = 'results_simple'
DATAFILE = 'final_data.csv'
USE_TEXT_COL = 'short_text'
VOCABULARY_SIZE = 15000
EMBEDDING_DIM = 200
WINDOW_SIZE_HALF = 2  # Default size of the windows. This is the number of target words on each side of the context word.
BATCH_SIZE = 16
NUM_NEGATIVE_EXAMPLES = 64
VALIDATION_SIZE = 16
VALIDATION_WINDOW = 100


class word2vec(object):
    def __init__(self, data, vocabulary_size=VOCABULARY_SIZE, embedding_dim=EMBEDDING_DIM, batch_size=BATCH_SIZE,
                 window_size_half=WINDOW_SIZE_HALF, num_negative_examples=NUM_NEGATIVE_EXAMPLES,
                 valid_size=VALIDATION_SIZE, valid_window=VALIDATION_WINDOW):
        self.x_ids = data
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.window_size_half = window_size_half
        self.batch_size = batch_size
        self.num_negative_examples = num_negative_examples
        self.valid_size = valid_size
        self.valid_window = valid_window

        # initialize batch vectors
        self.batch_x = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        self.batch_y = np.ndarray(shape=(self.batch_size, self.window_size_half * 2), dtype=np.int32)

        # Initialize counters for the batch generation
        self.sentence_counter = 0
        self.word_counter = 0
        self.epoch_counter = 0

        self.create_network()

    def create_network(self):
        # Create input variables
        self.input_x = tf.placeholder(tf.int32, [self.batch_size], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [self.batch_size, self.window_size_half * 2], name='input_y')

        # Create Embedding: [vocabulary_size, embedding_dim]
        with tf.name_scope('embedding'):
            init_width = 0.5 / self.embedding_dim
            self.embedding = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_dim], -init_width, init_width), name="emb")
            self.embed = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # Construct the variables for the NCE loss
        with tf.name_scope('output_layer'):
            # Create output weights transposed
            self.out_w_t = tf.Variable(tf.zeros([self.vocabulary_size, self.embedding_dim]), name="out_w_t")

            # Create output bias
            self.out_b = tf.Variable(tf.zeros([self.vocabulary_size]), name="out_b")

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.out_w_t,
                    biases=self.out_b,
                    labels=self.input_y,
                    inputs=self.embed,
                    num_sampled=self.num_negative_examples,
                    num_classes=self.vocabulary_size))

        # Store values of loss for tensorboard
        tf.summary.scalar('loss', loss)

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
        valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size // 2))
        valid_examples = np.append(valid_examples,
                                   random.sample(range(not_usual - self.valid_window, not_usual), self.valid_size // 2))

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
        normalized_embedding = self.embedding / norm
        valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        self.similarity = tf.matmul(valid_embedding, normalized_embedding, transpose_b=True)

    def generate_training_batch(self):
        '''
        This method generates the next training batch.
        :return: x_train and y_train. x_train will contain the context words to show at the input of the network.
        y_train will contain all the words to predict for each context word.
        '''

        index_word = 0
        index_sent = 0
        creating_batch = True
        i = 0
        while creating_batch:
            if index_word - self.window_size_half >= 0:
                if index_word + self.window_size_half < len(self.x_ids[index_sent]):
                    self.batch_x[i] = self.x_ids[index_sent][index_word]
                    self.batch_y[i, 0:self.window_size_half] = self.x_ids[index_sent][
                                                               i - self.window_size_half:index_word]
                    self.batch_y[i, 0:self.window_size_half] = self.x_ids[index_sent][
                                                               index_word + 1:index_word + self.window_size_half + 1]
                    i += 1
                    index_word += 1
                else:
                    index_word = 0
                    index_sent += 1
                    if index_sent >= len(self.x_ids):
                        index_sent = 0
                        self.epoch_counter += 1
            else:
                index_word += 1

            if i >= self.batch_x.shape[0]:
                creating_batch = False


def prepare_data(datafile=DATAFILE, use_text_col=USE_TEXT_COL, output_path=OUTPUT_FOLDER,
                 vocabulary_size=VOCABULARY_SIZE, store_vocabulary=True):
    # Gather data
    data = pd.read_csv(datafile)
    sentences = list(data[use_text_col])

    # Generate vocabulary
    vocabulary = Tokenizer(num_words=vocabulary_size, filters='!"%&()*,-./:;?@[\\]^_`{|}~\t\n')
    vocabulary.fit_on_texts(sentences)
    x_ids = vocabulary.texts_to_sequences(sentences)

    # Save vocabulary
    if store_vocabulary:
        with open(output_path + '/' + 'vocabulary', "wb") as f:
            pickle.dump(vocabulary.word_index, f, protocol=2)

    return x_ids, vocabulary


def main(_):
    # Create folder if it doesn't exist
    pau_utils.create_folder(OUTPUT_FOLDER + '/')

    x_train, _ = prepare_data()


if __name__ == "__main__":
    tf.app.run()
