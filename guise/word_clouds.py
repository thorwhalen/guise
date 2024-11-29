"""Word cloud generation with wordcloud.WordCloud."""

import re
from typing import Mapping, Callable, Union, Sequence, Any, Dict
from collections import Counter
from functools import partial
from operator import methodcaller

from i2 import Sig
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

from guise.util import (
    Text,
    Word,
    Words,
    TextToWords,
    Weight,
    WordWeights,
    WordsToWeights,
    WordsSpec,
    DFLT_STR_TO_WORD,
    DFLT_WORDS_TO_WEIGHTS,
)


DFLT_WORD_CLOUD_KWARGS: dict = dict()  # empty means use wordcloud defaults


def _probably_a_word_weight_dict(x) -> bool:
    return isinstance(x, dict) and isinstance(next(iter(x.values())), Weight)


def _apply_single_word_weighting_func_to_words(
    word_weighting_func: Callable[[Word], Weight], words: Words
) -> WordWeights:
    return dict(zip(words, map(word_weighting_func, words)))


def vectorize_word_weighting_func(
    word_weighting_func: Callable[[Word], Weight]
) -> WordsToWeights:
    return partial(_apply_single_word_weighting_func_to_words, word_weighting_func)


def _self_recovering_word_weights_calculation(
    words_to_weights: WordsToWeights, words: Words
) -> WordWeights:
    """
    Apply words_to_weights to words, but recover from most common mis-definitions.

    words_to_weights is supposed to be a words-list to word-weights dict function,
    but a user might specify a single-word-to-single-weight function, so we'll try
    to detect that and apply it to each word in the list of words to get the
    dict of word weights.
    """
    try:
        # Normal case: words are a list of strings
        word_weights = words_to_weights(words)
        if isinstance(word_weights, pd.Series):
            word_weights = word_weights.to_dict()
        # the words_to_weights could have workedm but if it didn't return a
        # {word: weight, ...} dict, we're going to assume it's a single-word-to-single-weight
        # and apply it to each word in the list
        if not _probably_a_word_weight_dict(word_weights):
            vectorized_word_weighting = vectorize_word_weighting_func(words_to_weights)
            word_weights = vectorized_word_weighting(words)
    except Exception as e:
        raise ValueError(
            "Your words_to_weights function failed when applied to your words list. \n"
            "Most common mistake is specifying a words_to_weights function as a "
            "single-word-to-single-weight function. \nIf that's what you have, please "
            "specify `words_to_weights=guise.vectorize_word_weighting_func(your_func)` "
            "instead. \n"
            f"Your words_to_weights was {words_to_weights} and {words[:3]=}.\n"
            f"Error was {e}."
        )
    return word_weights


def words_to_word_weights_dict(
    words: WordsSpec, str_to_words: TextToWords, words_to_weights: WordsToWeights
) -> WordWeights:
    if isinstance(str_to_words, str):
        # Use the provided callable to tokenize the input string into words
        regular_expression = str_to_words
        str_to_words = re.compile(regular_expression).findall
    if isinstance(words, str):
        # Use the provided callable to tokenize the input string into words
        string = words
        words = list(str_to_words(string))

    if _probably_a_word_weight_dict(words):
        # We already have a {word: frequency, ...} mapping
        return words
    else:
        # Use the provided callable to generate word weights
        return _self_recovering_word_weights_calculation(words_to_weights, words)


@(Sig(WordCloud).inject_into_keyword_variadic)
def word_cloud(
    words=None,
    save_filepath: str = None,
    *,
    str_to_words: TextToWords = DFLT_STR_TO_WORD,
    words_to_weights: WordsToWeights = DFLT_WORDS_TO_WEIGHTS,
    wc_decoder: Callable = lambda x: x,
    stopwords=None,
    **word_cloud_kwargs,
):
    """
    Create a word cloud from a list of words.

    The words can be a list of strings or a mapping from words to weights.

    The word cloud is created by applying the word_cloud function to the words.

    The word_cloud_kwargs are passed to the WordCloud constructor.

    Args:

    words: A list of words or a mapping from words to weights. If None, a partial
    function is returned with the desired parameters.

    save_filepath: The path to save the word cloud image to. If None, the word cloud
    image is not saved.

    str_to_words: A function that takes a text string and returns a list of words. By
    default, it uses the default regular expression to find words.

    words_to_weights: A function that takes a list of words and returns a mapping from
    words to weights. By default, it uses the Counter function to count the frequency of
    each word.

    wc_decoder: A function that takes a word cloud and returns the desired output. By
    default, it returns the image of the word cloud.

    stopwords: A set of words to ignore when creating the word cloud. By default, it uses
    the default set of stopwords from the wordcloud package.


    """
    if words is None:
        # just return a partial function with the desired parameters
        return partial(
            word_cloud,
            save_filepath=save_filepath,
            str_to_words=str_to_words,
            words_to_weights=words_to_weights,
            wc_decoder=wc_decoder,
            **word_cloud_kwargs,
        )

    word_weights = words_to_word_weights_dict(words, str_to_words, words_to_weights)

    word_cloud_kwargs = dict(
        DFLT_WORD_CLOUD_KWARGS, **word_cloud_kwargs, stopwords=stopwords
    )
    wc = WordCloud(**word_cloud_kwargs)

    # stopwords doesn't seem to work when using fit_words, so we have to do the work
    # ourselves:
    if stopwords is not None:
        word_weights = {
            word: weight
            for word, weight in word_weights.items()
            if word not in stopwords
        }
    wc.fit_words(word_weights)
    if save_filepath is not None:
        wc.to_file(save_filepath)
    return wc_decoder(wc)


@(Sig(WordCloud).inject_into_keyword_variadic)
def word_cloud_store(
    text_store: Mapping[Any, Text],
    wc_decoder: Callable = lambda wc: wc.to_image(),  # methodcaller('to_image'),
    *,
    str_to_words: TextToWords = DFLT_STR_TO_WORD,
    words_to_weights: WordsToWeights = DFLT_WORDS_TO_WEIGHTS,
    stopwords=None,
    **word_cloud_kwargs,
) -> Mapping[Any, WordCloud]:
    """
    Create a word cloud store from a text store.

    The text store is a mapping from keys to text strings. The word cloud store is a
    mapping from keys to word clouds.

    The word cloud store is created by applying the word_cloud function to each text
    in the text store.

    The word_cloud_kwargs are passed to the word_cloud function.

    The wc_decoder is a function that takes a word cloud and returns the desired
    output. By default, it returns the image of the word cloud.

    The str_to_words is a function that takes a text string and returns a list of words.
    By default, it uses the default regular expression to find words.

    The words_to_weights is a function that takes a list of words and returns a mapping
    from words to weights. By default, it uses the Counter function to count the
    frequency of each word.

    The stopwords is a set of words to ignore when creating the word cloud. By default,
    it uses the default set of stopwords from the wordcloud package.

    """
    from dol import wrap_kvs, Pipe

    _word_cloud = partial(
        word_cloud,
        str_to_words=str_to_words,
        words_to_weights=words_to_weights,
        wc_decoder=wc_decoder,
        stopwords=stopwords,
        **word_cloud_kwargs,
    )
    return wrap_kvs(text_store, value_decoder=_word_cloud)
