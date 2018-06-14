import tensorflow as tf
from word2vec_nba import word2vec
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.contrib.tensorboard.plugins import projector
import os
import pickle
import numpy as np
import pandas as pd
import pau_utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from collections import Counter


class Vocab(object):
    pass


folder = 'results/final_1/'
filename = 'initial_data'
EMBEDDING_DIM = 200

with open(folder + filename, "rb") as f:
    x_ids, vocab, reverse_vocabulary = pickle.load(f)

vocabulary = Vocab()
vocabulary.word_index = vocab
w2v = word2vec(x_ids, vocabulary, reverse_vocabulary, embedding_dim=EMBEDDING_DIM)

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
    #words_embed = w2v.final_embeddings
    embedding_dict = {}
    for i in indexes_words:
        embedding_dict[reverse_vocabulary[i]] = words_embed[i-1,:]

    # Get most similar
    word = '3-of-3'
    mat = pairwise_distances(w2v.final_embeddings,embedding_dict[word].reshape(1,-1))
    sorted_indexes = np.argsort(mat[:,0])[0:9]
    print("10 most similars of word {}: ".format(word))
    for i in sorted_indexes:
        print(reverse_vocabulary[i])

    # Analogy relations
    word1 = 'stephen'
    word2 = 'james'
    word3 = 'lebron'

    embedding_analogy = embedding_dict[word1] + (embedding_dict[word2] - embedding_dict[word3])

    mat_analogy = pairwise_distances(w2v.final_embeddings, embedding_analogy.reshape(1, -1))
    sorted_indexes = np.argsort(mat_analogy[:, 0])[0:9]
    print("10 most similars of analogy {}+{}-{} = : ".format(word1,word2,word3))
    for i in sorted_indexes:
        print(reverse_vocabulary[i])


    # Tests
    word1 = 'cavs'
    word2 = 'lebron'

    embedding_analogy2 = embedding_dict[word1] - embedding_dict[word2]

    mat_analogy2 = pairwise_distances(w2v.final_embeddings, embedding_analogy2.reshape(1, -1))
    sorted_indexes2 = np.argsort(mat_analogy2[:, 0])[0:9]
    print("10 most similars of analogy {}-{} = : ".format(word1,word2))
    for i in sorted_indexes2:
        print(reverse_vocabulary[i])


    #########################
    # Create projector for players and teams
    #########################

    projector_name = 'projector'

    # First find list of players surnames
    data = pd.read_csv('final_data2.csv')
    player_complete_names = list(data['player'])
    team_complete_names = list(data['team'])
    player_names_all = []
    team_names_all = []
    for name in set(player_complete_names):
        player_names_all.append(name.split(' ')[1].strip().lower())
        possible_teams = [team_complete_names[i] for i, x in enumerate(player_complete_names) if x == name]
        possible_t = Counter(possible_teams)
        team_names_all.append(possible_t.keys()[0].strip().lower())

    subsample = True # Get a subset of them, otherwise the visualization is too complex
    if subsample:
        # Lets do it for only a subset of teams
        #teams = ['cavaliers','warriors','mavericks','thunder','rockets']
        #teams = ['boston', 'spurs','warriors', 'magic','heat','grizzlies']
        teams = ['celtics', 'spurs', 'warriors', 'magic', 'heat']
        #teams = ['celtics', 'spurs', 'warriors', 'magic', 'jazz']
        player_names = [x for i,x in enumerate(player_names_all) if team_names_all[i] in teams]
        team_names = [x for i, x in enumerate(team_names_all) if team_names_all[i] in teams]
    else:
        player_names = player_names_all
        team_names = team_names_all


    final_vectors = np.zeros([len(player_names),EMBEDDING_DIM])
    for i in range(len(player_names)):
        final_vectors[i,:] = embedding_dict[player_names[i]]

    pau_utils.create_folder(folder + projector_name)
    # Write corresponding labels for the embeddings.
    with open(folder + projector_name+ '/metadata_sub.tsv', 'w') as f:
        for i in xrange(len(player_names)):
            if i == 0:
                f.write('Player'+'\t'+'Team'+ '\n')
                f.write(player_names[i] + '\t' + team_names[i] + '\n')
            else:
                f.write(player_names[i] + '\t' + team_names[i] + '\n')

    # Try projector creation
    embedding = tf.Variable(tf.stack(final_vectors, axis=0), trainable=False, name='embedding_sub')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(folder + projector_name, sess.graph)

    # Add embedding tensorboard visualization. Need tensorflow version
    # >= 0.12.0RC0
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'embedding_sub:0'
    embed.metadata_path = 'metadata_sub.tsv'
    projector.visualize_embeddings(writer, config)

    # We save the embeddings for TensorBoard
    saver.save(sess, os.path.join(folder, projector_name+ '/sub_model.ckpt'))


    # Plot TSNE
    tsne = TSNE(
        perplexity=40, n_components=2, init='random', n_iter=1000, method='barnes_hut',random_state = 12)
    low_dim_embs = tsne.fit_transform(final_vectors)

    possible_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    color_team = {name:possible_colors[i] for i,name in enumerate(set(team_names))}
    fig = plt.figure(figsize=(18, 18))  # in inches
    #fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, label in enumerate(team_names):
        x, y = low_dim_embs[i, :]
        #ax.scatter(x, y, c=color_team[label],cmap=cmap,norm=norm,edgecolors='none',label=label)
        ax.scatter(x, y, c = color_team[label], label=label)
        ax.annotate(
            player_names[i],
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    leg = plt.legend(set(team_names))
    for marker,lab in zip(leg.legendHandles,set(team_names)):
        marker.set_color(color_team[lab])

    plt.title('T-SNE')
    plt.savefig(folder+'T-SNE.png')

    # Plot PCA
    pca = PCA(n_components=2)
    low_dim_embs2 = pca.fit_transform(final_vectors)

    possible_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    color_team = {name: possible_colors[i] for i, name in enumerate(set(team_names))}
    fig = plt.figure(figsize=(18, 18))  # in inches
    # fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, label in enumerate(team_names):
        x, y = low_dim_embs2[i, :]
        # ax.scatter(x, y, c=color_team[label],cmap=cmap,norm=norm,edgecolors='none',label=label)
        ax.scatter(x, y, c=color_team[label], label=label)
        ax.annotate(
            player_names[i],
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    leg = plt.legend(set(team_names))
    for marker, lab in zip(leg.legendHandles, set(team_names)):
        marker.set_color(color_team[lab])
    plt.title('PCA')
    plt.savefig(folder+'PCA.png')
