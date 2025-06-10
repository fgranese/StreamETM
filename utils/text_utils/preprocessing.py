import re
import string
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = {
    "french": set(stopwords.words("french")),
    "english": set(stopwords.words("english"))
}

def remove_too_freq_words(texts, max_freq=0.2):
    # Normalize the texts (convert to lowercase, remove punctuation)
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with spaces
        return text

    texts = [normalize_text(text) for text in texts]

    # Calculate word frequencies
    word_freq = Counter(word for text in texts for word in text.split())
    total_words = sum(word_freq.values())

    # Identify words to remove
    words_to_remove = {word for word, freq in word_freq.items() if freq / total_words > max_freq}

    # Remove frequent words from the texts
    filtered_texts = [' '.join(word for word in text.split() if word not in words_to_remove) for text in texts]

    # List of too frequent words
    too_freq_words = [word for word, freq in word_freq.items() if freq / total_words > max_freq]

    return filtered_texts, too_freq_words

def remove_low_freq_words(texts, min_freq):
    word_freq = Counter(word for text in texts for word in text.split())
    words_to_keep = {word for word, freq in word_freq.items() if freq > min_freq}
    filtered_texts = [' '.join(word for word in text.split() if word in words_to_keep) for text in texts]
    rare_words = [word for word, freq in word_freq.items() if freq == 1]
    return filtered_texts, rare_words

def lemmatize_texts(texts, model_name, verbose=True):
    if verbose:
        print(f"Lemmatizing texts using {model_name} model...")
    nlp = spacy.load(model_name, exclude=["parser", "ner"])
    lemmatized_texts = [
        ' '.join(token.lemma_ for token in doc)
        for doc in nlp.pipe(texts, n_process=1, batch_size=512)
    ]
    return lemmatized_texts

def remove_small_texts(texts, min_words=10):
    # Filter the list of texts by counting words in each entry
    filtered_texts = [text for text in texts if len(text.split()) >= min_words]
    return filtered_texts

def preprocess_text(texts, language="english", verbose=True):
    if language not in STOPWORDS:
        raise ValueError("Language not supported. Please choose 'french' or 'english'.")
    
    stopwords_lang = STOPWORDS[language]
    
    if language == "french":
        model_name = "fr_core_news_lg"
    elif language == "english":
        model_name = "en_core_web_lg"
    else:
        raise ValueError("Language not supported. Please choose 'french' or 'english'.")
    
    # Lemmatize texts
    texts = lemmatize_texts(texts, model_name, verbose=verbose)

    # Preprocess text: lowercase, remove punctuation, tokenize, and filter stopwords
    processed_texts = []
    for text in texts:
        text = text.replace("'", " ").lower()
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stopwords_lang and len(word) > 2]
        processed_texts.append(' '.join(filtered_tokens))

    # Remove low-frequency words
    processed_texts, rare_words = remove_low_freq_words(processed_texts, min_freq=1)
    
     # Remove too frequent words
    word_count = Counter()

    # Count word frequency in each document (document frequency)
    for text in processed_texts:
        unique_words = set(text.split())
        word_count.update(unique_words)

    # Calculate the threshold for too frequent words
    total_docs = len(processed_texts)
    max_freq=0.7
    threshold = max_freq * total_docs

    # Get words that appear in more than the specified percentage of documents
    too_freq_words = {word for word, count in word_count.items() if count > threshold}

    # Remove the too frequent words from the processed texts
    final_processed_texts = []
    for text in processed_texts:
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in too_freq_words]
        final_processed_texts.append(' '.join(filtered_tokens))

    if verbose:
        print(f"Number of rare words: {len(rare_words)}")
        print(f"Number of too frequent words: {len(too_freq_words)}")

    return processed_texts

