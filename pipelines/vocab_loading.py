import numpy as np
import os

VOCAB_MODEL = {
    'glove' : 'load_glove_vocab_embeddings',
}

def load_glove_vocab_embeddings(save_flag: bool=True):
    import gensim.downloader as api

    # Load the model
    # https://code.google.com/archive/p/word2vec/
    glove_model = api.load("glove-wiki-gigaword-300")

    # show info about models and datasets available in gensim
    api.info()

    vocab = [word for word in glove_model.key_to_index if word.isalpha() and word.islower() and len(word) > 2]
    embeddings = np.array([glove_model.get_vector(word) for word in glove_model.key_to_index if word.isalpha() and word.islower() and len(word) > 2])


    print('Vocab size:', len(vocab))
    print('Embeddings shape:', embeddings.shape)
    print('Vocab:', vocab[:100])

    os.makedirs('data/vocab/', exist_ok=True)
    if save_flag:
        np.save('data/vocab/vocab_glove.npy', vocab)
        np.save('data/vocab/glove_embeddings.npy', embeddings)

    return vocab, embeddings


def create_vocab(vocab_name: str='glove', save_flag: bool=True, **params):
    print(vocab_name)
    return eval(VOCAB_MODEL[vocab_name])(save_flag=save_flag, **params)


if __name__ == '__main__':
    model_type = 'glove'
    create_vocab(model_type, save_flag=True)
