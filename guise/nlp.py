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
