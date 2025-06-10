import logging
from logging import getLogger

import numpy as np

logger = getLogger(__name__)

def log_optimal_transport_plan(ot_plan,
                               model1_topics,
                               model2_topics,
                               decimals=4):
    """
    Logs the Optimal Transport plan along with topic words from both models.

    Parameters:
        ot_plan (ndarray): The Optimal Transport plan matrix.
        model1_topics (list of lists): Top words for each topic in Model 1.
        model2_topics (list of lists): Top words for each topic in Model 2.
        decimals (int): Number of decimal places to display for OT plan values.
    """
    ot_plan_transposed = ot_plan.T
    ot_plan_rounded = np.round(ot_plan_transposed, decimals=decimals)

    col_width = max(len(f"{val:.{decimals}f}") for row in ot_plan_rounded for val in row) + 2
    total_width = max(col_width, max(len(' '.join(topic[:3])) for topic in model1_topics))

    header = " " * (total_width + 20) + " | ".join(
        [f"Topic {i + 1}".center(col_width) for i in range(len(model2_topics))]
    )
    logging.info(header)
    logging.info("-" * len(header))

    for idx, row in enumerate(ot_plan_rounded):
        if idx >= len(model1_topics):
            logging.warning(f"Index {idx} is out of bounds for model1_topics")
            continue

        row_values = " | ".join([f"{val:.{decimals}f}".ljust(col_width) for val in row])
        topic_words = ' '.join(model1_topics[idx][:3]).rjust(total_width)
        logging.info(f"Topic {idx + 1} (Model 1): {topic_words} | {row_values}")

    logging.info("-" * len(header))

    for i, words in enumerate(model2_topics):
        topic_words = ' '.join(words[:3]).ljust(col_width * len(model2_topics))
        logging.info(f"Topic {i + 1} (Model 2): {topic_words}")

def log_model_topics(model,
                     proportions):
    from utils.text_utils.text_utils import get_top_words_for_topic
    beta = model.get_beta(numpy=True)
    vocab = model.vocab
    logging.info("### Found Topics in the Model ###")
    for idx in range(beta.shape[0]):
        topic_words = get_top_words_for_topic(beta, vocab, num_topic=idx)
        logging.info(f"Topic {idx + 1}: Proportion = {proportions[idx]:.4f} - {topic_words}")

def log_topic_topics_filter(proportions, epsilon, topics_to_keep):

    logger.info(f"Mean Proportion: {proportions.mean()}")
    logger.info(f"Epsilon: {epsilon}")

    logger.info(f"length of proportions: {proportions.shape}")
    logger.info(f"Topics to keep length: {topics_to_keep.shape}")