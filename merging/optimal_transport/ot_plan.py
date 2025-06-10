import ot
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import cdist

def compute_ot_plan(m1_embeddings,
                    m2_embeddings,
                    dist='cosine', ot_fun='ot.unbalanced.mm_unbalanced'):

    print(dist, ot_fun)

    num_topics1 = m1_embeddings.shape[0]  # old model
    num_topics2 = m2_embeddings.shape[0]  # new model

    a = np.full(num_topics2, 1 / num_topics2)  # new
    b = np.full(num_topics1, 1 / num_topics1)  # old

    if dist == 'cosine':
        cost_matrix = 1 - cosine_similarity(m2_embeddings, m1_embeddings)
    else:
        cost_matrix = cdist(m2_embeddings, m1_embeddings, metric=dist)

    cost_matrix = cost_matrix / cost_matrix.max()

    ot_plan = eval(ot_fun)(a, b, cost_matrix, reg_m=0.09, reg=0, div='kl')
    return ot_plan