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

OUTPUT_FOLDER = 'results_simple'
DATAFILE = 'final_data.csv'
USE_TEXT_COL = 'short_text'
VOCABULARY_SIZE = 15000


def generate_batches(x_ids, batch_size, num_skips, skip_windows):


    return x_train, y_train


def prepare_data(datafile=DATAFILE, use_text_col=USE_TEXT_COL, output_path=OUTPUT_FOLDER, store_vocabulary=True):
    # Gather data
    data = pd.read_csv(datafile)
    sentences = list(data[use_text_col])

    # Generate vocabulary
    vocabulary = Tokenizer(num_words=VOCABULARY_SIZE, filters='!"%&()*,-./:;?@[\\]^_`{|}~\t\n')
    vocabulary.fit_on_texts(sentences)
    x_ids = vocabulary.texts_to_sequences(sentences)

    # Save vocabulary
    with open(output_path + '/' + vocabulary, "wb") as f:
        pickle.dump(vocabulary.word_index, f, protocol=2)

    # Generate batches
    x_train, y_train = generate_batches(x_ids)




def main(_):
    # Create folder if it doesn't exist
    pau_utils.create_folder(OUTPUT_FOLDER + '/')

    dataset, vocabulary = prepare_data()


if __name__ == "__main__":
    tf.app.run()
