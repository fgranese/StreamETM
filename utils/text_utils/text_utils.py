import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def create_document_term_matrix(data, vocab, min_df=1, max_df=0.90):
    vectorizer = CountVectorizer(vocabulary=vocab, min_df=min_df, max_df=max_df)
    return vectorizer.fit_transform(data)



def get_top_words_for_topic(beta, vocab, num_topic, nWords=10):
    """
    Get the top words for a specific topic.
    """
    K = beta.shape[0]
    assert 0 <= num_topic < K, f"Invalid num_topic: {num_topic}. Must be between 0 and {K-1}."
    ord = np.argsort(beta[num_topic, :])[::-1]
    top_words = np.array(vocab)[ord[:nWords]]
    return top_words.tolist()

def get_top_words(beta, vocab, num_top_words, verbose=False):
    """
    Get the top words for each topic.
    """
    topic_str_list = []
    for i, topic_dist in enumerate(beta):
        topic_words = np.argsort(topic_dist)[::-1][:num_top_words]
        top_words = np.array(vocab)[topic_words]
        if verbose:
            logging.info(f"Topic {i}: {top_words}")
        topic_str_list.append(top_words)
    return topic_str_list

def filter_topics_by_proportion(theta,
                                epsilon=0.05):
    """
    Filters topics based on their proportion in the theta matrix.

    Parameters:
        theta (ndarray): Document-topic distribution matrix (num_docs x num_topics).
        epsilon (float): Threshold for minimum topic proportion.

    Returns:
        tuple: (topics_to_keep (ndarray), proportions (ndarray))
            topics_to_keep: Boolean array indicating which topics to keep.
            proportions: Proportion of each topic in the corpus.
    """
    dominant_topic = np.argmax(theta, axis=1)
    one_hot_matrix = np.zeros_like(theta)
    one_hot_matrix[np.arange(theta.shape[0]), dominant_topic] = 1

    topic_counts = one_hot_matrix.sum(axis=0)
    total_counts = one_hot_matrix.sum()
    proportions = topic_counts / total_counts

    ## adaptative epsilon
    epsilon = epsilon * proportions.mean()


    topics_to_keep = proportions >= epsilon

    from merging.logging_merge import log_topic_topics_filter
    log_topic_topics_filter(proportions, epsilon, topics_to_keep)


    return topics_to_keep, proportions