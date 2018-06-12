"""
Embedding for NBA news from 2017-2018 season
Author: Pau Bramon

This code was created following the Tensorflow tutorial.
"""

import tensorflow as tf
import pau_utils
import pandas as pd
import pickle
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

OUTPUT_FOLDER = 'results_simple'
DATAFILE = 'final_data.csv'
USE_TEXT_COL = 'short_text'
VOCABULARY_SIZE = 15000
SKIP_WINDOW = 2


class word2vec(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []


def generate_training_data(x_ids, skip_windows):
    num_instances = 0
    skip_sentence = []
    for ind, s in enumerate(x_ids):
        incre = len(s) / (skip_windows * 2 + 1)
        num_instances += incre
        if incre == 0:
            skip_sentence.append(ind)

    x_train = np.ndarray(shape=(num_instances, 1), dtype=np.int32)
    y_train = np.ndarray(shape=(num_instances, skip_windows * 2), dtype=np.int32)
    span = skip_windows * 2 + 1
    index_word = 0
    index_sent = 0

    for i in range(x_train.shape[0]):
        x_train[i] = x_ids[index_sent][index_word + skip_windows]
        y_train[i, 0:skip_windows] = x_ids[index_sent][index_word:index_word + skip_windows]
        y_train[i, skip_windows:] = x_ids[index_sent][index_word + skip_windows + 1:index_word + skip_windows * 2 + 1]
        index_word += span
        if index_word + span > len(x_ids[index_sent]):
            index_word = 0
            searching_next = True
            while(searching_next):
                index_sent += 1
                if index_sent not in skip_sentence:
                    searching_next = False

    return x_train, y_train


def prepare_data(datafile=DATAFILE, use_text_col=USE_TEXT_COL, output_path=OUTPUT_FOLDER, skip_window=SKIP_WINDOW,
                 vocabulary_size = VOCABULARY_SIZE, store_vocabulary=True):
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

    # Generate batches
    x_train, y_train = generate_training_data(x_ids, skip_window)

    return x_train, y_train, vocabulary


def main(_):
    # Create folder if it doesn't exist
    pau_utils.create_folder(OUTPUT_FOLDER + '/')

    x_train,y_train, _ = prepare_data()


if __name__ == "__main__":
    tf.app.run()
