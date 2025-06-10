import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd

from topic_models.trainer import BasicTrainer
import topic_models.etm as etm
from merging.merge import merge_etm_models
import utils.general_utils as gen_ut
import utils.text_utils.text_utils as text_ut
from utils.text_utils.preprocessing import preprocess_text

from ocpd.ocpd import install_ocp_package, run_ocpd_analysis, filter_change_points, store_change_points


def merge_models(old_model, new_model, dtm, device, params):
    with torch.no_grad():
        theta = new_model.get_theta(dtm, numpy=True)
    return merge_etm_models(old_model.cpu(), new_model.cpu(), current_theta=theta, **params)


def process_data_chunk(chunk, vocab, embeddings, device, model_params, trainer_params, previous_model=None,
                       refine=False):
    dtm = text_ut.create_document_term_matrix(chunk, vocab)
    alpha = previous_model.topic_embeddings.cpu() if previous_model else None
    num_topics = previous_model.num_topics if previous_model else model_params['num_topics']

    model = etm.ETM(
        num_topics=num_topics,
        vocab_size=len(vocab),
        hidden_size=model_params['hidden_size'],
        embed_size=embeddings.shape[1],
        embeddings=embeddings,
        train_embeddings=model_params['train_embeddings'],
        enc_drop=model_params['enc_drop'],
        vocab=vocab,
        alpha=alpha
    ).to(device)

    trainer = BasicTrainer(model=model, dataset=dtm, **trainer_params)
    trainer.train(refine)
    return model, dtm


def train_topic_model_online_base(config: dict, documents: list, time_step: str):
    vocab_type = config['vocab']
    embedding_type = config['embeddings']

    documents = preprocess_text(documents, language=config['language'])
    vocab = np.load(f'./data/vocab/vocab_{vocab_type}.npy')[:config['model_params']['max_vocab_size']]
    embeddings = np.load(f"./data/vocab/{embedding_type}_embeddings.npy")[:config['model_params']['max_vocab_size']]

    print(vocab_type, vocab.shape, embedding_type, embeddings.shape)

    run_folder = gen_ut.create_run_directory(config['config_file'])
    os.makedirs(run_folder, exist_ok=True)

    device = torch.device(config['trainer_params']['device'] if torch.cuda.is_available() else 'cpu')

    previous_model_path = os.path.join(run_folder, 'model.pkl')
    previous_model = pickle.load(open(previous_model_path, 'rb')) if os.path.exists(previous_model_path) else None

    print(previous_model_path, run_folder)

    if previous_model:
        model, dtm = process_data_chunk(
            documents, vocab, embeddings, device,
            config['model_params'], config['trainer_params']
        )

        dtm_tensor = torch.from_numpy(dtm.toarray()).float().to(device)

        model = merge_models(
            previous_model, model, dtm_tensor, device,
            config['merge_params']
        )
        model.num_topics = previous_model.num_topics
        model, dtm = process_data_chunk(
            documents, vocab, embeddings, device,
            config['model_params'], config['trainer_params'], model, refine=True
        )
    else:
        model, dtm = process_data_chunk(
            documents, vocab, embeddings, device,
            config['model_params'], config['trainer_params']
        )
        dtm_tensor = torch.from_numpy(dtm.toarray()).float().to(device)

    with open(previous_model_path, 'wb') as f:
        pickle.dump(model, f)

    theta_df = compute_theta(model, dtm_tensor, time_step, documents)
    topic_over_time, topics_representation = compute_topic_distributions_representations(model, theta_df, run_folder,
                                                                                         vocab, time_step)

    threshold = config['ocpd_params']['threshold']
    change_points = online_change_point_detection(threshold, topic_over_time, run_folder, time_step)
    return model, dtm_tensor, change_points


def compute_theta(model, dtm_tensor, time_step, documents):
    theta = model.get_theta(dtm_tensor, numpy=True)
    theta_df = pd.DataFrame(theta).assign(
        topic_pred=lambda df: df.idxmax(axis=1),
        time_bin=time_step
    )

    theta_df['documents'] = documents
    theta_df = theta_df[['time_bin', 'topic_pred', 'documents'] + [col for col in theta_df.columns if
                                                                   col not in ['time_bin', 'topic_pred', 'documents']]]
    theta_df.columns = theta_df.columns.astype(str)

    return theta_df


def compute_topic_distributions_representations(model, theta_df, run_folder, vocab, time_step):
    doc_topic_distribution_path = os.path.join(run_folder, 'doc_topic_distribution.csv')

    if os.path.exists(doc_topic_distribution_path):
        doc_topic_distribution = pd.read_csv(doc_topic_distribution_path)
    else:
        doc_topic_distribution = pd.DataFrame()

    doc_topic_distribution = pd.concat([doc_topic_distribution, theta_df], ignore_index=True)

    top_words = text_ut.get_top_words(
        model.get_beta(numpy=True), vocab, num_top_words=10, verbose=True
    )
    topics_representation = [
        {"time_bin": time_step, "topic": idx, "top_words": words}
        for idx, words in enumerate(top_words)
    ]

    topic_over_time = doc_topic_distribution.groupby(['time_bin', 'topic_pred']).size().unstack().fillna(0)
    gen_ut.save_to_file(topic_over_time, run_folder, 'topic_over_time.csv')
    gen_ut.save_to_file(doc_topic_distribution, run_folder, 'doc_topic_distribution.csv')
    gen_ut.save_to_file(pd.DataFrame(topics_representation), run_folder, f'topics_representation_{time_step}.csv')
    gen_ut.save_to_file(pd.DataFrame(topics_representation), run_folder, f'topics_representation.csv')

    return topic_over_time, topics_representation


def online_change_point_detection(threshold, topic_over_time, run_folder, time_step):

    install_ocp_package()
    topic_ids = topic_over_time.columns.tolist()
    topic_over_time = topic_over_time.reset_index(drop=True)
    proportions_csv = os.path.join(run_folder, 'topic_over_time.csv')
    ocpd_csv = os.path.join(run_folder, 'ocpd.csv')
    change_points = run_ocpd_analysis(proportions_csv, threshold)
    change_points = filter_change_points(change_points, topic_over_time)
    store_change_points(change_points, time_step, ocpd_csv, topic_ids)

    return change_points
