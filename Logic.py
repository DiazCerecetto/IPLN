import re
import spacy
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import pandas as pd
import numpy as np
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score


############################################################
#                   GLOBAL FUNCTIONS
############################################################

def load_fasttext(filename, limit=None):
    """
    Loads FastText word vectors (in Word2Vec format) from the specified file.
    
    :param filename: Path to the FastText file in Word2Vec format.
    :param limit: If not None, limits the number of word vectors to load.
    :return: A Gensim KeyedVectors object.
    """
    model = KeyedVectors.load_word2vec_format(filename, limit=limit)
    return model


def train_model_with_random_search(model, param_distributions, X_train, y_train, 
                                   cv=3, n_iter=50, random_state=123):
    """
    Performs a RandomizedSearchCV on a given model with specified parameter distributions.
    
    :param model: The model/estimator to be optimized.
    :param param_distributions: Dictionary with parameters names (string)
                               as keys and distributions or lists of parameters to try.
    :param X_train: Training feature set.
    :param y_train: Training labels.
    :param cv: Number of cross-validation folds.
    :param n_iter: Number of parameter settings sampled.
    :param random_state: Seed for reproducibility.
    :return: Best estimator after performing randomized search.
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='f1_macro',
        cv=cv,
        random_state=random_state,
        n_jobs=10
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        random_search.fit(X_train, y_train)
    return random_search.best_estimator_


def evaluate_model(model, X_dev, y_dev):
    """
    Evaluates a model using macro F1-score.
    
    :param model: Trained model.
    :param X_dev: Development/validation feature set.
    :param y_dev: Ground truth labels for the development set.
    :return: Macro F1-score.
    """
    y_pred = model.predict(X_dev)
    return f1_score(y_dev, y_pred, average='macro')


def build_custom_embeddings(fasttext_model, corpus_processed, top_n=0):
    """
    Builds a custom dictionary of word embeddings from the given FastText model,
    based on the token frequencies in `corpus_processed`.

    :param fasttext_model: A loaded FastText model in Gensim format.
    :param corpus_processed: List of [id, text, label] where `text` is already preprocessed.
    :param top_n: If > 0, only the top_n most frequent tokens are used; otherwise all tokens.
    :return: A dictionary {token: embedding_vector}.
    """
    all_tokens = []
    for row in corpus_processed:
        all_tokens.extend(row[1].split())

    freq_counter = Counter(all_tokens)
    freq_df = pd.DataFrame({'token': freq_counter.keys(), 'count': freq_counter.values()})
    freq_df.sort_values(by='count', ascending=False, inplace=True)

    if top_n > 0:
        words_to_use = list(freq_df[:top_n].token.values)
    else:
        words_to_use = list(freq_df.token.values)

    custom_dict = {}
    for w in words_to_use:
        if w in fasttext_model.key_to_index:
            custom_dict[w] = fasttext_model[w]

    print(f"Custom dictionary size: {len(custom_dict)}")
    return custom_dict


############################################################
#                   LINGUISTIC ANALYZER
############################################################

class LinguisticAnalyzer:
    """
    Class to perform linguistic analysis using spaCy.
    """

    def __init__(self, model_name='es_dep_news_trf'):
        """
        Initializes the class by loading the specified spaCy model.

        :param model_name: spaCy model for linguistic analysis (Spanish by default).
        """
        self.nlp = spacy.load(model_name)

    def analyze_text(self, text):
        """
        Performs linguistic analysis on the given text using spaCy.

        :param text: Text to analyze.
        :return: spaCy Doc object with the analysis.
        """
        return self.nlp(text)

    def show_analysis(self, text):
        """
        Prints a basic analysis of the text, including word, POS tag, 
        dependency type, and the head of the syntactic tree.

        :param text: Text to analyze.
        """
        doc = self.analyze_text(text)
        for token in doc:
            print(token.text, token.pos_, token.dep_, token.head)

    ######################## Word-Level Analysis ########################

    def pos_tags(self, text):
        """
        Returns a list of POS tags for each token in the text.

        :param text: Text to analyze.
        :return: List of POS tags.
        """
        doc = self.analyze_text(text)
        return [token.pos_ for token in doc]

    def lemmas(self, text):
        """
        Returns a list of lemmas for each token in the text.

        :param text: Text to analyze.
        :return: List of lemmas.
        """
        doc = self.analyze_text(text)
        return [token.lemma_ for token in doc]

    def word_pos_lemma(self, text):
        """
        Returns a list of 3-tuples (word, POS, lemma) for each token in the text,
        including punctuation.

        :param text: Text to analyze.
        :return: List of (word, POS, lemma) tuples.
        """
        tokens = re.findall(r"[\w']+|[.,:!?;]", text)  # Includes punctuation
        tags = self.pos_tags(text)
        lemmas_ = self.lemmas(text)
        return list(zip(tokens, tags, lemmas_))

    ######################## Sentence-Level Analysis ########################

    def get_roots(self, text):
        """
        Returns the roots of the syntactic trees in the text (tokens with dependency 'ROOT').

        :param text: Text to analyze.
        :return: List of root tokens.
        """
        doc = self.analyze_text(text)
        return [token.text for token in doc if token.dep_ == 'ROOT']

    def get_major_phrases(self, text):
        """
        Returns a list of major nominal and prepositional phrases in the text, 
        including their syntactic function.

        :param text: Text to analyze.
        :return: List of [phrase, description] pairs.
        """
        doc = self.analyze_text(text)
        results = []
        roots = [token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]

        for root in roots:
            for token in root.children:
                phrase = ''.join([t.text_with_ws for t in token.subtree])
                is_prepositional = any(elem.pos_ == "ADP" for elem in token.subtree)
                phrase_type = 'PP: ' if is_prepositional else 'NP: '

                if not self.phrase_in_set(phrase, results):
                    if token.dep_ == "nsubj":
                        results.append([phrase, phrase_type + "Subject " + token.dep_])
                    elif token.dep_ == "obl":
                        results.append([phrase, phrase_type + "Oblique " + token.dep_])
                    elif token.dep_ == "iobj":
                        results.append([phrase, phrase_type + "Indirect Object " + token.dep_])
                    elif token.dep_ in ["obj"]:
                        self._analyze_object(token, results, phrase_type, root)
                    elif token.dep_ == "expl:pv":
                        results.append([phrase, phrase_type + "Reflexive Pronoun " + token.dep_])

        return self._filter_major_phrases(results)

    def phrase_in_set(self, phrase, phrase_list):
        """
        Checks if a given phrase is already in the stored list.

        :param phrase: Phrase to check.
        :param phrase_list: List of existing phrases.
        :return: True if the phrase is present, False otherwise.
        """
        return any(saved_phrase == phrase for saved_phrase, _ in phrase_list)

    def _analyze_object(self, token, results, phrase_type, root):
        """
        Analyzes and classifies an object token as direct, indirect,
        or 'Complemento de Régimen' (a Spanish-specific grammar concept).
        
        :param token: The object token to analyze.
        :param results: The list to store results.
        :param phrase_type: 'PP' or 'NP' prefix based on the phrase type.
        :param root: The root verb of the sentence.
        """
        for sub_token in token.subtree:
            if sub_token.dep_ == "case":
                if sub_token.text in ["de", "con", "por"]:
                    results.append([token.text, phrase_type + "Regimen Complement " + token.dep_])
                elif sub_token.text == "a":
                    is_complement = any(
                        child.dep_ == "expl:pv" and child.pos_ == "PRON" for child in root.children
                    )
                    if is_complement:
                        results.append([token.text, phrase_type + "Regimen Complement " + token.dep_])
                    else:
                        results.append([token.text, phrase_type + "Indirect Object " + token.dep_])
                break
            else:
                # Simple heuristic for direct/indirect object
                pronouns = ["le", "la", "lo", "los", "las", "les"]
                if token.text.lower() in pronouns:
                    results.append([token.text, phrase_type + "Indirect Object " + token.dep_])
                else:
                    results.append([token.text, phrase_type + "Direct Object " + token.dep_])
                break

    def _filter_major_phrases(self, phrase_list):
        """
        Filters the list of phrases to remove nested duplicates and keep only major phrases.

        :param phrase_list: List of phrases.
        :return: Filtered list of phrases.
        """
        for phrase_info in phrase_list[:]:
            phrase_text, _ = phrase_info
            for phrase_info2 in phrase_list:
                phrase_text2, _ = phrase_info2
                if phrase_text != phrase_text2 and phrase_text2 in phrase_text:
                    phrase_list.remove(phrase_info2)
                    break
        return phrase_list


############################################################
#                   PREPROCESSING
############################################################

class Preprocessing:
    """
    Class for text preprocessing (cleaning, normalization, vectorization).
    """

    def __init__(self):
        pass

    def replace_url(self, text):
        """
        Removes URLs from the text.

        :param text: Input text.
        :return: (updated_text, changed) - modified text and a bool if a change occurred.
        """
        updated_text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!\*\(\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text
        )
        changed = (updated_text != text)
        return updated_text, changed

    def replace_user_mention(self, text):
        """
        Removes user mentions from the text (e.g., @username).

        :param text: Input text.
        :return: (updated_text, changed)
        """
        updated_text = re.sub(r"@(\w+)", "", text)
        changed = (updated_text != text)
        return updated_text, changed

    def replace_abbreviations(self, text):
        """
        Replaces common Spanish abbreviations with their full forms.

        :param text: Input text.
        :return: (updated_text, changed)
        """
        abbreviations = {
            r'[\s^][xX][qQ][\s$]': ' porque ',
            r'[\s^][pP](\s)*[qQ][\s$]': ' porque ',
            r'porq': ' porque ',
            r'[\s^][xX][\s$]': ' por ',
            r'[\s^][qQ][\s$]': ' que ',
            r'[\s^][kK][\s$]': ' que ',
            r'[\s^][bB][nN][\s$]': ' bien ',
            r'[\s^][tT][mM][bB][\s$]': ' tambien ',
            r'[\s^][rR][tT][\s$]': ' ',
            r'[\s^][aA][cC][eE][Ss][\s$]': ' haces ',
            r'[\s^][bB][bB][\?*\s$]': ' bebé ',
            r'[\s^][vV][sS][\s$]': ' versus ',
            r'[\s^][cC][\s$]': ' se ',
            r'[\s^]\+[\s$]': ' mas ',
            r'[\s^][dD][\s$]': ' de ',
            r'[\s^][dD][lL][\s$]': ' del ',
            r'[\s^][tT][aA][\s$]': ' está ',
            r'[\s^][pP][aA][\s$]': ' para ',
            r'[\s^][pP][sS][\?*.*,*\s$]': ' pues ',
            r'[\s^][mM][\s$]': ' me ',
            r'[\s^][cC][sS][mM][\s$]': ' insult ',
            r'[\s^][gG]ral[\s.$]': ' general ',
            r'[\s^][dD][rR][.\s$]': ' doctor ',
            r'[\s^][mM][gG][\s$]': ' me gusta '
        }

        updated_text = text
        for pattern, replacement in abbreviations.items():
            updated_text = re.sub(pattern, replacement, updated_text)

        changed = (updated_text != text)
        return updated_text, changed

    def strip_accents(self, text):
        """
        Removes Spanish accents from the text (á, é, í, ó, ú).

        :param text: Input text.
        :return: Text without accents.
        """
        return re.sub(
            r'[áéíóúÁÉÍÓÚ]',
            lambda m: {
                'á': 'a','é': 'e','í': 'i','ó': 'o','ú': 'u',
                'Á': 'A','É': 'E','Í': 'I','Ó': 'O','Ú': 'U',
            }.get(m.group(), m.group()),
            text
        )

    def remove_extra_spaces(self, text):
        """
        Collapses multiple consecutive spaces into a single space.

        :param text: Input text.
        :return: Text with only single spaces.
        """
        return re.sub(r'\s+', ' ', text)

    def remove_stopwords(self, text, stopwords):
        """
        Removes stopwords from the text.

        :param text: Input text.
        :param stopwords: A set of stopwords to remove.
        :return: Text without the given stopwords.
        """
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w not in stopwords]
        return " ".join(filtered)

    def preprocess_tweet(self, tweet, stopwords=[]):
        """
        Applies multiple preprocessing steps to a tweet.

        :param tweet: Original tweet text.
        :param stopwords: Set of stopwords to remove.
        :return: Cleaned and processed tweet text.
        """
        if not tweet:
            return ''

        text = tweet.lower()
        text = self.strip_accents(text)
        text, _ = self.replace_user_mention(text)
        text, _ = self.replace_url(text)
        text, _ = self.replace_abbreviations(text)
        text = self.remove_extra_spaces(text)
        text = self.remove_stopwords(text, stopwords)

        return text

    def preprocess_corpus(self, corpus, stopwords=[]):
        """
        Preprocesses an entire corpus where each row is [id, tweet, label].

        :param corpus: List of [id, tweet_text, label].
        :param stopwords: Set of stopwords to remove.
        :return: Processed corpus with the same structure [id, cleaned_text, label].
        """
        new_corpus = []
        for row in corpus:
            tweet_id, tweet, label = row
            processed_tweet = self.preprocess_tweet(tweet, stopwords)
            new_corpus.append([tweet_id, processed_tweet, label])
        return new_corpus

    def tweet_to_mean_vector(self, tokens, emb_dict, neg_lexicon=[], pos_lexicon=[]):
        """
        Converts a list of tokens into a mean vector of embeddings (size 300), 
        plus 2 additional features for the counts of negative and positive words.

        :param tokens: List of tokens in the tweet.
        :param emb_dict: Dictionary {word: embedding_vector}.
        :param neg_lexicon: List/set of negative words.
        :param pos_lexicon: List/set of positive words.
        :return: Numpy array of shape (302,) = (300-dim mean vector + 2 counters).
        """
        emb_size = 300
        vecs = []
        neg_count = 0
        pos_count = 0

        for w in tokens:
            if w in neg_lexicon:
                neg_count += 1
            if w in pos_lexicon:
                pos_count += 1
            if w in emb_dict:
                vecs.append(emb_dict[w])

        if len(vecs) > 0:
            mean_vec = np.mean(vecs, axis=0)
        else:
            mean_vec = np.zeros(emb_size)

        return np.concatenate([mean_vec, [neg_count, pos_count]])

    def tweet_to_concat_vector(self, tokens, emb_dict, neg_lexicon=[], pos_lexicon=[], max_length=10):
        """
        Converts a list of tokens into a concatenated vector of size (max_length * 300) + 2, 
        by stacking up to 'max_length' embeddings in a row and adding counters.

        :param tokens: List of tokens.
        :param emb_dict: Dictionary {word: embedding_vector}.
        :param neg_lexicon: Set of negative words.
        :param pos_lexicon: Set of positive words.
        :param max_length: Max tokens to concatenate.
        :return: Numpy array of shape ((max_length * 300) + 2,).
        """
        emb_size = 300
        arr = np.zeros(max_length * emb_size + 2)
        neg_count = 0
        pos_count = 0
        gathered = []

        for w in tokens:
            if w in neg_lexicon:
                neg_count += 1
            if w in pos_lexicon:
                pos_count += 1
            if w in emb_dict:
                gathered.append(emb_dict[w])

        idx = 0
        for v in gathered[:max_length]:
            arr[idx : idx + emb_size] = v
            idx += emb_size

        arr[max_length * emb_size] = neg_count
        arr[max_length * emb_size + 1] = pos_count
        return arr


############################################################
#                   LSTM UTILS
############################################################

class LSTMUtils:
    """
    Utility class for building and handling LSTM models 
    with either mean-vector or concatenated-sequence approaches.
    """
    def __init__(self):
        self.preprocessor = Preprocessing()

    def encode_label(self, y_list):
        """
        Encodes a list of string labels into numeric integers:
        'P' -> 0, 'N' -> 1, 'NONE' -> 2.

        :param y_list: List of labels (e.g., ['P', 'N', 'NONE']).
        :return: Numpy array of encoded labels (0, 1, 2).
        """
        label_map = {'P': 0, 'N': 1, 'NONE': 2}
        return np.array([label_map[x] for x in y_list])

    def one_hot_3classes(self, y_encoded):
        """
        One-hot encodes an array of labels [0,1,2] into a shape (N, 3).

        :param y_encoded: Array of integer labels in {0,1,2}.
        :return: One-hot-encoded array of shape (len(y_encoded), 3).
        """
        out = np.zeros((len(y_encoded), 3))
        for i, val in enumerate(y_encoded):
            out[i, val] = 1
        return out

    def build_lstm_model_for_mean_vector(self, input_dim=302):
        """
        Builds an LSTM model that takes a single time step with dimension (302).

        :param input_dim: Dimension of the input vector (default 302).
        :return: Compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Input(shape=(1, input_dim)))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_lstm_model_for_concat_sequence(self, max_length=10, embedding_dim=302):
        """
        Builds an LSTM model that takes sequences of shape (max_length, 302).

        :param max_length: Maximum sequence length.
        :param embedding_dim: Number of features per time step (default 302).
        :return: Compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Input(shape=(max_length, embedding_dim)))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def tweet_to_concat_sequence(self, tokens, emb_dict, neg_lexicon=[], pos_lexicon=[], max_length=10):
        """
        Returns a 2D array of shape (max_length, 302) for LSTM:
          - 300-dim embedding
          - 2 additional features (neg_count, pos_count) 
            repeated at each time step (example heuristic).
        
        :param tokens: List of tokens.
        :param emb_dict: Dictionary of embeddings {word: vector}.
        :param neg_lexicon: Set of negative words.
        :param pos_lexicon: Set of positive words.
        :param max_length: Maximum time steps (tokens).
        :return: Numpy array of shape (max_length, 302).
        """
        emb_size = 300
        arr = np.zeros((max_length, emb_size + 2), dtype=np.float32)

        neg_count = 0
        pos_count = 0
        gathered = []

        # Count negative and positive words; gather embeddings
        for w in tokens:
            if w in neg_lexicon:
                neg_count += 1
            if w in pos_lexicon:
                pos_count += 1
            if w in emb_dict:
                gathered.append(emb_dict[w])

        # Fill up to max_length
        for i in range(min(len(gathered), max_length)):
            arr[i, :emb_size] = gathered[i]
            arr[i, emb_size]   = neg_count
            arr[i, emb_size+1] = pos_count

        return arr

    def build_mean_vector_dataset(self, processed_data, emb_dict, neg_lexicon=[], pos_lexicon=[]):
        """
        Converts each tweet into a (302,) vector (300-dim mean + 2 counters)
        and reshapes it to (1, 302) for a single time step in the LSTM.

        :param processed_data: List of [id, cleaned_text, label].
        :param emb_dict: Dictionary {token: embedding_vector}.
        :param neg_lexicon: List/set of negative words.
        :param pos_lexicon: List/set of positive words.
        :return: (X_3d, y) where X_3d has shape (num_samples, 1, 302).
        """
        X = []
        y = []
        for row in processed_data:
            tokens = row[1].split()
            mean_vec = self.preprocessor.tweet_to_mean_vector(tokens, emb_dict, neg_lexicon, pos_lexicon)
            X.append(mean_vec)
            y.append(row[2])

        X = np.array(X)
        y = self.encode_label(y)
        X_3d = np.expand_dims(X, axis=1)  # (num_samples, 1, 302)
        return X_3d, y

    def build_concat_sequence_dataset(self, processed_data, emb_dict, neg_lexicon=[], 
                                      pos_lexicon=[], max_length=10):
        """
        Converts each tweet into a sequence of shape (max_length, 302).

        :param processed_data: List of [id, cleaned_text, label].
        :param emb_dict: Dictionary {token: embedding_vector}.
        :param neg_lexicon: Set of negative words.
        :param pos_lexicon: Set of positive words.
        :param max_length: Max sequence length to consider.
        :return: (X, y) where X has shape (num_samples, max_length, 302).
        """
        X_list = []
        y_list = []
        for row in processed_data:
            tokens = row[1].split()
            seq = self.tweet_to_concat_sequence(tokens, emb_dict, neg_lexicon, pos_lexicon, max_length=max_length)
            X_list.append(seq)
            y_list.append(row[2])

        X = np.array(X_list)  # shape (num_samples, max_length, 302)
        y = self.encode_label(y_list)
        return X, y
