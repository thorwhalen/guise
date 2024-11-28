"""Natural language processing tools"""

from typing import Mapping, Any, Iterable, Union, Callable
from collections import Counter, defaultdict

from html2text import html2text
from idiom import word_frequencies
from lexis import Lemmas

from guise.util import STOPWORDS

Word = str
Words = Iterable[Word]
WordCount = Mapping[Word, int]
WordStats = Mapping[Word, Any]
WordMap = Mapping[Word, Word]
StemmerSpec = Union[Callable, str, None]

# ---------------------------------------------------------------------------------------
# tfidf scoring


_cache = {}


DFLT_INCLUDE_TERMS = 'most_frequent_english'
DFLT_EXCLUDE_TERMS = STOPWORDS


def get_include_terms(include=DFLT_INCLUDE_TERMS) -> Words:
    """allowed_terms = 'most_frequent_english' or 'wordnet_lemmas'"""
    include = include or 'most_frequent_english'
    if include == 'most_frequent_english':
        if 'most_frequent_english' not in _cache:
            _cache['most_frequent_english'] = set(dict(word_frequencies()))
        include = _cache['most_frequent_english']
    elif include == 'wordnet_lemmas':
        if 'wordnet_lemmas' not in _cache:
            _cache['wordnet_lemmas'] = set(Lemmas())
        include = _cache['wordnet_lemmas']
    return include


def html_tokens(html: str, include=DFLT_INCLUDE_TERMS, exclude=STOPWORDS) -> Words:
    toks = map(str.strip, html2text(html).split(' '))
    if include is not None:
        include = get_include_terms(include)
        include = set(include) - (exclude or {})
        return filter(include.__contains__, toks)
    elif exclude is not None:
        return filter(lambda x: x not in exclude, toks)
    else:
        return toks


def term_filtered_html2text(
    html, include=DFLT_INCLUDE_TERMS, exclude=DFLT_EXCLUDE_TERMS, sep=' ; '
) -> str:
    """allowed_terms = 'most_frequent_english' or 'wordnet_lemmas'"""
    return sep.join(html_tokens(html, include, exclude))


def stem_based_word_mapping(
    words: Words, stemmer: StemmerSpec = 'lancaster'
) -> WordMap:
    """Returns a dict of {word: replace_with_word, ...} mapping based on stemming.

    stemming is great to reduce the number of words by replacing several


    The words (or word counts) are stemmed, then for each stem, the word with the
    highest count gets to represent the others.

    >>> _words = ['happy', 'happier', 'happiest', 'happily', 'sad', 'sadly']
    >>> word_count = Counter(_words)
    >>> stem_based_word_mapping(word_count)  # doctest: +NORMALIZE_WHITESPACE
    {'happy': 'happy',
     'happier': 'happy',
     'happiest': 'happiest',
     'happily': 'happy',
     'sad': 'sad',
     'sadly': 'sad'}
    """
    if not callable(stemmer):
        from nltk.stem import PorterStemmer
        from nltk.stem import LancasterStemmer

        if stemmer is None:
            stemmer = lambda word: word
        elif isinstance(stemmer, str):
            if stemmer.lower().startswith('porter'):
                stemmer = PorterStemmer().stem
            elif stemmer.lower().startswith('lancaster'):
                stemmer = LancasterStemmer().stem
            else:
                raise ValueError(f'Unknown stemmer value: {stemmer}')
        else:
            raise TypeError(f'Unknown stemmer type: {stemmer}')

    if isinstance(words, Mapping):
        word_count = words
    else:
        word_count = Counter(words)

    word_of_stem = defaultdict(list)
    for word, count in word_count.items():
        word_of_stem[stemmer(word)].append((count, word))

    def sort_key(count_and_word):
        count, word = count_and_word
        # favor the highest count, if not, the shortest word, if not lexicographic
        return -count, len(word), word

    word_of_stem = {
        stem: sorted(count_and_words, key=sort_key)[0][1]
        for stem, count_and_words in word_of_stem.items()
    }

    return {word: word_of_stem[stemmer(word)] for word in word_count}


def map_words_of_word_count(word_count: WordCount, word_map: WordMap) -> WordCount:
    """Recompute a word count based on a (complete) mapping of it's words (keys)

    >>> assert (dict(
    ... map_words_of_word_count(
    ...     {'one': 1, 'ones': 2, '1': 3, 'two': 4},
    ...     {'one': 'one', 'ones': 'one', '1': 'one', 'two': 'two'}))
    ... == {'one': 6, 'two': 4}
    ... )

    """
    c = Counter()
    for word, count in word_count.items():
        c.update({word_map[word]: count})
    return c


def stem_based_word_count(
    words: Words, stemmer: StemmerSpec = 'lancaster'
) -> WordCount:
    """From an iterable of words, get a WordCount mapping which uses a stem-based map

    >>> words = ['happy', 'happier', 'happiest', 'happily', 'sad', 'sadly']
    >>> assert (dict(stem_based_word_count(words))
    ... == {'happy': 3, 'happiest': 1, 'sad': 2})

    """
    word_count = Counter(words)
    word_map = stem_based_word_mapping(word_count, stemmer)
    return map_words_of_word_count(word_count, word_map)


# --------------------------------------------------------------------------------------
# 2024

from typing import Mapping
import numpy as np
import pandas as pd


class TFIDFCalculator:
    """
    A TF-IDF calculator that computes TF-IDF scores for documents based on a corpus.

    Parameters
    ----------
    corpus_counts : pandas.Series
        Series with index as words and values as counts in the corpus.
    idf_type : str, optional
        The type of IDF calculation to use. Options are:
        - 'standard': IDF = log(N / (df))
        - 'smooth': IDF = log(1 + N / (df))
        - 'probabilistic': IDF = log((N - df) / df)
        - 'frequency': IDF = log((total_counts_corpus + 1) / (corpus_counts + 1))
        Default is 'frequency'.
    stop_words : set or list, optional
        A set or list of stop words to exclude from calculations.
    custom_idf : dict or pandas.Series, optional
        A mapping from words to custom IDF values.

    Attributes
    ----------
    idf_ : pandas.Series
        Inverse Document Frequency (IDF) values computed from the corpus.
    default_idf_ : float
        Default IDF value for words not found in the corpus.

    Examples
    --------
    >>> import pandas as pd
    >>> # Sample corpus word counts
    >>> corpus_counts = pd.Series({
    ...     'word1': 300,
    ...     'word2': 200,
    ...     'word4': 500,
    ...     'trump': 100
    ... })
    >>> stop_words = {'the', 'and', 'is'}
    >>> custom_idf = {'trump': 5.0}
    >>> tfidf_calculator = TFIDFCalculator(corpus_counts,
    ...                                    idf_type='standard',
    ...                                    stop_words=stop_words,
    ...                                    custom_idf=custom_idf)
    >>> # Sample document word counts
    >>> doc_counts = pd.Series({
    ...     'word1': 3,
    ...     'word2': 2,
    ...     'trump': 4,
    ...     'harris': 2,
    ...     'the': 10
    ... })
    >>> tfidf_scores = tfidf_calculator(doc_counts)
    >>> print(tfidf_scores)  # doctest: +ELLIPSIS
    word1    -1.17...
    word2    -0.71...
    trump     1.81...
    harris    0.90...
    dtype: float64
    """

    def __init__(
        self, corpus_counts, idf_type='frequency', stop_words=None, custom_idf=None
    ):
        self.stop_words = set(stop_words) if stop_words else set()
        self.corpus_counts = corpus_counts.astype(float).copy()
        # Remove stop words from corpus counts
        self.corpus_counts.drop(labels=self.stop_words, errors='ignore', inplace=True)
        self.total_counts_corpus = self.corpus_counts.sum()
        self.N = len(self.corpus_counts)  # Number of unique terms in the corpus

        # Compute IDF values
        self.idf_type = idf_type
        self.idf_ = self._compute_idf()

        default_idf = self.idf_.max()

        # Apply custom IDF values if provided
        if custom_idf:
            if isinstance(custom_idf, (list, tuple, set)):
                custom_idf = {word: default_idf for word in custom_idf}
            custom_idf_series = pd.Series(custom_idf)
            self.idf_.update(custom_idf_series)

        # Compute default IDF for words not in corpus
        self.default_idf_ = self.idf_.max()

    def _compute_idf(self):
        if self.idf_type == 'standard':
            # IDF = log(N / df)
            idf = np.log(self.N / (self.corpus_counts + 1e-10))
        elif self.idf_type == 'smooth':
            # IDF = log(1 + N / df)
            idf = np.log(1 + self.N / (self.corpus_counts + 1e-10))
        elif self.idf_type == 'probabilistic':
            # IDF = log((N - df) / df)
            idf = np.log(
                (self.N - self.corpus_counts + 1e-10) / (self.corpus_counts + 1e-10)
            )
        elif self.idf_type == 'frequency':
            # IDF = log((total_counts_corpus + 1) / (corpus_counts + 1))
            idf = np.log((self.total_counts_corpus + 1) / (self.corpus_counts + 1))
        else:
            raise ValueError(
                f"Unknown idf_type '{self.idf_type}'. Choose from 'standard', 'smooth', 'probabilistic', 'frequency'."
            )

        return idf

    def __call__(self, doc_counts):
        """
        Computes the TF-IDF scores for a document based on word counts.

        Parameters
        ----------
        doc_counts : pandas.Series
            Series with index as words and values as counts in the document.

        Returns
        -------
        tfidf : pandas.Series
            TF-IDF scores for each word in the document.
        """
        if not isinstance(doc_counts, pd.Series):
            if isinstance(doc_counts, Mapping):
                doc_counts = pd.Series(doc_counts)
            elif isinstance(doc_counts, (list, tuple)):
                words = doc_counts
                doc_counts = pd.Series(Counter(words))
            elif isinstance(doc_counts, str):
                words = doc_counts.split()
                doc_counts = pd.Series(Counter(words))

        doc_counts = doc_counts.astype(float).copy()
        # Remove stop words from document counts
        doc_counts.drop(labels=self.stop_words, errors='ignore', inplace=True)

        total_counts_doc = doc_counts.sum()
        # Compute Term Frequency (TF)
        tf = doc_counts / total_counts_doc
        # Get IDF values for words in the document
        idf = self.idf_.reindex(doc_counts.index)
        # For words not in the corpus, assign default IDF value
        idf = idf.fillna(self.default_idf_)
        # Compute TF-IDF
        tfidf = tf * idf
        return tfidf
