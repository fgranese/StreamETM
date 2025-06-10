import numpy as np
import torch
import torch.nn as nn

import merging.logging_merge as logging_merge
from merging.optimal_transport.ot_plan import compute_ot_plan
from topic_models.etm import ETM
from utils.text_utils import text_utils as text_ut

def  merge_topic_embeddings(m1,
                           m2,
                           ot_plan,
                           weighted=False):
    matching_indices = np.argmax(ot_plan, axis=1)
    matching_weights = np.max(ot_plan, axis=1)

    if weighted:
        matching_weights = matching_weights / (1 / len(matching_weights))
    merged_topics = m1.copy()
    for idx_m2, (idx_m1, weight) in enumerate(zip(matching_indices, matching_weights)):
        if weight > 1e-3:
            if weighted:
                merged_topics[idx_m1] = (merged_topics[idx_m1] * (1 - weight) + m2[idx_m2] * weight) / 2
            else:
                merged_topics[idx_m1] = (merged_topics[idx_m1] + m2[idx_m2]) #/2
        else:
            merged_topics = np.vstack([merged_topics, m2[idx_m2]])
    return merged_topics


def merge_etm_models(model1,
                     model2,
                     current_theta,
                     epsilon=0.05,
                     distance='cosine'):

    m1_embeddings = model1.get_topic_embeddings(numpy=True)  # old model
    m2_embeddings = model2.get_topic_embeddings(numpy=True)  # new model

    topics_to_keep, proportions = text_ut.filter_topics_by_proportion(current_theta, epsilon)
    m2_filtered_embeddings = m2_embeddings[topics_to_keep]

    model2.topic_embeddings = nn.Parameter(
        torch.tensor(m2_filtered_embeddings, dtype=torch.float32)
    )

    logging_merge.log_model_topics(model2, proportions)

    beta1 = model1.get_beta(numpy=True)
    model1_topics = [text_ut.get_top_words_for_topic(beta1, model1.vocab, num_topic=idx) for idx in
        range(m1_embeddings.shape[0])]
    beta2 = model2.get_beta(numpy=True)
    model2_topics = [text_ut.get_top_words_for_topic(beta2, model2.vocab, num_topic=idx) for idx in
        range(m2_filtered_embeddings.shape[0])]
    ot_plan = compute_ot_plan(m1_embeddings, m2_filtered_embeddings, dist=distance)

    logging_merge.log_optimal_transport_plan(ot_plan, model1_topics, model2_topics)

    merged_embeddings = merge_topic_embeddings(m1_embeddings, m2_filtered_embeddings, ot_plan, weighted=False)

    merged_model = ETM(
        num_topics=merged_embeddings.shape[0], vocab_size=model1.vocab_size, hidden_size=model1.hidden_size,
        embed_size=model1.embed_size, embeddings=model1.word_embeddings.detach().cpu().numpy(),
        train_embeddings=model1.word_embeddings.requires_grad, enc_drop=model1.enc_drop, vocab=model1.vocab
    )

    merged_model.topic_embeddings = nn.Parameter(
        torch.tensor(merged_embeddings, dtype=torch.float32)
    )
    return merged_model
