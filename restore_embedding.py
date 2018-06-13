import tensorflow as tf
from word2vec_nba import word2vec
from sklearn.metrics.pairwise import pairwise_distances
import pickle
import numpy as np


class Vocab(object):
    pass


folder = 'results_simple_2nd/'
filename = 'initial_data'

with open(folder + filename, "rb") as f:
    x_ids, vocab, reverse_vocabulary = pickle.load(f)

vocabulary = Vocab()
vocabulary.word_index = vocab
w2v = word2vec(x_ids, vocabulary, reverse_vocabulary)

# Create network
w2v.create_network()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, folder + "model.ckpt")
    print("Model restored.")
    w2v.final_embeddings = w2v.normalized_embedding.eval()

    # # Find neighbours to
    # find_neighbours_to = 472  # Lebron
    #
    # sim = w2v.similarity.eval()
    # valid_word = w2v.reverse_vocabulary[find_neighbours_to]
    # top_k = 10  # number of nearest neighbors
    # nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    # log_str = 'Nearest to %s:' % valid_word
    # for k in xrange(top_k):
    #     close_word = w2v.reverse_vocabulary[nearest[k]]
    #     log_str = '%s %s,' % (log_str, close_word)
    # print(log_str)

    # Create dictionary 1:
    indexes_words = sorted(vocab.values())
    list_sorted_keys = [reverse_vocabulary[i] for i in indexes_words]
    ind_dataset = tf.constant(indexes_words, dtype=tf.int32)
    words_embedding = tf.nn.embedding_lookup(w2v.normalized_embedding, ind_dataset)
    words_embed = sess.run(words_embedding)
    embedding_dict = {}
    for i in indexes_words:
        embedding_dict[reverse_vocabulary[i]] = words_embed[i-1,:]

    # Get most similar
    word = 'cavs'
    mat = pairwise_distances(w2v.final_embeddings,embedding_dict[word].reshape(1,-1))
    sorted_indexes = np.argsort(mat[:,0])[0:10]
    print("10 most similars of word {}: ".format(word))
    for i in sorted_indexes:
        print(reverse_vocabulary[i])

    # Get relations
    word1 = 'cavs'
    word2 = 'james'
    word3 = 'curry'

    embedding_analogy = embedding_dict[word1] - embedding_dict[word2] + embedding_dict[word3]

    mat = pairwise_distances(w2v.final_embeddings, embedding_analogy.reshape(1, -1))
    sorted_indexes = np.argsort(mat[:, 0])[0:10]
    print("10 most similars of analogy {}-{}+{} = : ".format(word1,word2,word3))
    for i in sorted_indexes:
        print(reverse_vocabulary[i])