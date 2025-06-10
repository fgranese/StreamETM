import os
from os import listdir
from os.path import isfile, join
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import logging

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logging.getLogger('gensim').setLevel(logging.WARNING)


def calculate_topic_diversity(topics, topk=10):
    """
    Calculate the diversity of topics based on the top-k words.
    """
    if topk > len(topics[0]):
        raise Exception("Words in topics are less than " + str(topk), len(topics[0]), topics[0])
    unique_words = set()
    for topic in topics:
        unique_words = unique_words.union(set(topic[:topk]))
    puw = len(unique_words) / (topk * len(topics))
    return round(puw, 3)


def calculate_coherence(topics, texts, bigram_transformer=None, coherence='c_v'):
    """
    Calculate the coherence score for the given topics and texts.
    """
    if bigram_transformer:
        texts = [bigram_transformer[text] for text in texts]
    dictionary = Dictionary(texts)
    # corpus = [dictionary.doc2bow(token) for token in texts]
    coherence_model_lda = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence,
                                         processes=10)
    coherence_lda = coherence_model_lda.get_coherence()
    # logging.info(f"Coherence Score: {coherence_lda}")
    return coherence_lda


def map_topics_to_original(topics_original_tw: dict,
                           topics_predicted_tw: dict,
                           embedding_func: str = 'get_bert_embeddings',
                           glove_model=None, technique: str = 'cosine'):
    if (embedding_func == 'get_glove_embeddings') and (glove_model is None):
        glove_model = api.load("glove-wiki-gigaword-300")
    embeddings_predicted = {k: eval(embedding_func)(v, glove_model) for k, v in topics_predicted_tw.items()}
    embeddings_original = {k: eval(embedding_func)(v, glove_model) for k, v in topics_original_tw.items()}

    if technique == 'cosine':
        cosine_matrix = np.zeros((len(embeddings_predicted.keys()), len(embeddings_original.keys())))
        for k_pred, v_pred in embeddings_predicted.items():
            for k_org, v_org in embeddings_original.items():
                cosine_matrix[k_pred, k_org] = cosine_similarity(v_org, v_pred).max()

        max_values = np.max(cosine_matrix, axis=1)
        sum_values = np.sum(cosine_matrix, axis=1)
        concentration_ratios = max_values / sum_values
        sorted_indices = np.argsort(concentration_ratios)[::-1]
        topics_assigned = {}

        for row in sorted_indices[:cosine_matrix.shape[
            1]]:  # consider only the first cosine_matrix.shape[1] of most concentrated topics
            org_topic_map = -1
            topic_mappings_ = cosine_matrix.copy()
            while org_topic_map not in topics_assigned.values():
                org_topic_map = topic_mappings_[row, :].argmax()
                if org_topic_map not in topics_assigned.values():
                    topics_assigned[row] = org_topic_map

                else:
                    topic_mappings_[row, org_topic_map] = -np.inf
                    org_topic_map = -1

    return cosine_matrix, concentration_ratios, sorted_indices, topics_assigned


def get_most_important_keywords(topic_id, topics_representation_pd: pd.DataFrame, n_key_words=5,
                                flag_original_pd: bool = False, run_folder: str = ''):
    def filter_words(words, original=False):
        return [word for word in words if
                (len(word) <= 10 and len(word) > 3) and not any(char.isdigit() for char in word)]

    if not flag_original_pd:
        topics_representation_pd, _ = load_topic_representations_file(run_folder)
    topics_representation_pd = topics_representation_pd[topics_representation_pd.topic != -1]
    topics = topics_representation_pd[topics_representation_pd.topic == topic_id].top_words.values[:]

    if flag_original_pd:
        return topics
    else:
        topics = list(
            itertools.chain.from_iterable(
                np.array([t.strip("[]").replace("'", "").replace(",", "").split() for t in topics])
            )
        )
        topics = filter_words(topics)
        return list(zip(*Counter(topics).most_common(n_key_words)))[0]


def load_topic_representations_file(run_folder_path: str):
    file_names = np.array([f for f in listdir(run_folder_path) if isfile(join(run_folder_path, f))])
    topics_representation_file = file_names[
        np.flatnonzero(np.core.defchararray.find(file_names, 'topics_representation_') != -1)]
    topics_representation_pd = pd.concat(
        [pd.read_csv(os.path.join(run_folder_path, f)) for f in topics_representation_file])
    return topics_representation_pd, file_names


def plot_topics_over_time(topics: pd.DataFrame, change_points: pd.DataFrame, time_serie: pd.DataFrame,
                          flag_original_pd: bool = False,
                          path_plots_dir: str = '/user/fgranese/home/stream_etm/streamETM2/plots/',
                          file_name_plot: str = 'plot.png',
                          title: str = '', run_folder: str = '', rupture_points: list = [], bbox=(1, 0.5),
                          SMALL_SIZE=8, MEDIUM_SIZE=10, BIGGER_SIZE=12, k=1, figsize=(16, 10), topic_colors=None,
                          scenario='extreme'):
    import matplotlib.pyplot as plt
    topics = topics.sort_values(by='topic')
    plt.figure(figsize=figsize)

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plot_colors = {}

    offset_value = 0.1
    original_top_words = {}

    def get_cmap(n, name='Set1'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    cmap = get_cmap(len(topics['topic']))
    for topic, cp in enumerate(topics['topic']):
        print(cp)
        label = get_most_important_keywords(int(cp), topics, flag_original_pd=flag_original_pd, run_folder=run_folder)
        original_top_words[int(cp)] = eval(label.item()) if isinstance(label, np.ndarray) else label
        if topic_colors is None:
            if scenario == 'custom':
                if any(ext in label[:3] for ext in ['insurance', 'auto', 'brake', 'wheel', 'shift', 'drive', 'toyota']):
                    color = '#1f77b4'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['christian', 'bible', 'jesus', 'religion', 'truth', 'love']):
                    color = '#ff7f0e'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['mission', 'satellite', 'space', 'shuttle']):
                    color = '#2ca02c'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['baseball', 'sport']):
                    color = '#9467bd'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['medicine', 'muscle']):
                    color = '#d62728'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                else:
                    print(cmap(topic))
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=cmap(topic),
                                     linewidth=1, alpha=0.3)
            elif scenario == 'extreme':
                if any(ext in label[:3] for ext in ['software', 'scsi', 'chip', 'file']):
                    color = 'darkblue'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['security', 'wiretap', 'encryption']):
                    color = 'darkgreen'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in
                         ['jesus', 'christian', 'religion', 'christians', 'society', 'bible', 'church']):
                    color = 'mediumorchid'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['turkish', 'israeli', 'armenian', 'arab']):
                    color = 'orange'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                elif any(ext in label[:3] for ext in ['sale', 'forsale']):
                    color = 'red'
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=color,
                                     linewidth=4)
                else:
                    print(cmap(topic))
                    line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=cmap(topic),
                                     linewidth=1, alpha=0.3)
        else:
            line, = plt.plot(change_points['Datetime'], time_serie[str(cp)], label=label[:3], color=topic_colors[topic],
                             linewidth=4)
        plot_colors.update({topic: line.get_color()})

    # num_topics = len(topics['topic'])
    # offsets = np.linspace(-offset_value, offset_value, num_topics)
    # print(offsets, num_topics)

    alpha_value = 1  # Set transparency
    linestyles = ['--', '-.', ':']  # Different linestyles for distinction

    for i, topic in enumerate(topics['topic']):
        for j, cp in enumerate(change_points[str(topic)]):
            if cp and not np.isnan(cp):
                print('topic', i, 'cp at', j)
                offset_cp = j  # j + offsets[i]
                plt.axvline(
                    offset_cp, color=plot_colors.get(topic), linestyle=linestyles[i % len(linestyles)],
                    alpha=alpha_value
                )

    # for rupture in rupture_points:
    #     plt.axvline(x=rupture, color='b', linestyle='-', alpha=0.7)

    # print(k)
    legend = plt.legend(loc='center left', bbox_to_anchor=bbox, ncol=k)
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_linewidth(0.0)

    plt.xticks(rotation=90)
    plt.xlabel("Time")
    plt.ylabel("Topic Count")
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(os.path.join(path_plots_dir, file_name_plot), bbox_inches='tight')
    plt.show()
    return plt, original_top_words,


def load_files_from_path(ts_path, top_file, ch_file, tb_file):
    time_serie = pd.read_csv(ts_path).fillna(0)
    topics = pd.read_csv(top_file).fillna(0)
    topics = topics[topics.topic != -1]
    change_points = pd.read_csv(ch_file)
    # datetime = np.load(tb_file, allow_pickle=True)
    return time_serie, topics, change_points


def get_glove_embeddings(vocab,
                         glove_model=None):
    import gensim.downloader as api
    if glove_model is None:
        glove_model = api.load("glove-wiki-gigaword-300")

    most_similar = {'harddisk': 'hard-disk', 'forsale': 'sale', 'mousecom': 'mouse', 'configsys': 'config.sys',
                    'ultrastor': 'store', 'lekoff': 'lakoff', 'soc': 'society'}
    vectors = []
    for word in vocab:
        if word in most_similar.keys():
            vectors.append(glove_model.get_vector(word.replace(word, most_similar[word])))
        else:
            vectors.append(glove_model.get_vector(word))
    return np.asarray(vectors)
