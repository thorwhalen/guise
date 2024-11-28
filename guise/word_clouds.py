"""Word cloud generation with wordcloud.WordCloud."""

import re
from typing import Mapping, Callable, Union, Sequence, Any, Dict
from collections import Counter
from functools import partial
from operator import methodcaller

from i2 import Sig
from wordcloud import WordCloud

Text = str
Words = Sequence[Text]
TextToWords = Callable[[Text], Words]
WordWeights = Dict[str, Union[int, float]]
WordsToWeights = Callable[[Words], Mapping[Text, int]]
WordsSpec = Union[Words, WordWeights]

DFLT_WORD_CLOUD_KWARGS: dict = dict()  # empty means use wordcloud defaults
DFLT_STR_TO_WORD: TextToWords = re.compile(r'\w+').findall
DFLT_WORDS_TO_WEIGHTS: WordsToWeights = Counter


def _proabably_a_word_weight_dict(x) -> bool:
    return isinstance(x, dict) and isinstance(next(iter(x.values())), (int, float))


def _ensure_word_weights(
    words: WordsSpec, str_to_words: TextToWords, words_to_weights: WordsToWeights
) -> WordWeights:
    if isinstance(words, str):
        # Use the provided callable to tokenize the input string into words
        string = words
        words = list(str_to_words(string))
    if _proabably_a_word_weight_dict(words):
        # We already have a {word: frequency, ...} mapping
        weight_for_word = words
    else:
        # Use the provided callable to generate word weights
        weight_for_word = words_to_weights(words)
    return weight_for_word


@Sig(WordCloud).inject_into_keyword_variadic
def word_cloud(
    words,
    save_filepath: str = None,
    *,
    str_to_words: TextToWords = DFLT_STR_TO_WORD,
    words_to_weights: WordsToWeights = DFLT_WORDS_TO_WEIGHTS,
    **word_cloud_kwargs,
):
    weight_for_word = _ensure_word_weights(words, str_to_words, words_to_weights)

    word_cloud_kwargs = dict(DFLT_WORD_CLOUD_KWARGS, **word_cloud_kwargs)
    wc = WordCloud(**word_cloud_kwargs)
    wc.fit_words(weight_for_word)
    if save_filepath is not None:
        wc.to_file(save_filepath)
    return wc


def word_cloud_store(
    text_store: Mapping[Any, Text],
    wc_decoder: Callable = lambda wc: wc.to_image(),  # methodcaller('to_image'),
    *,
    str_to_words: TextToWords = DFLT_STR_TO_WORD,
    words_to_weights: WordsToWeights = DFLT_WORDS_TO_WEIGHTS,
    **word_cloud_kwargs,
) -> Mapping[Any, WordCloud]:
    from dol import wrap_kvs, Pipe

    _word_cloud = partial(
        word_cloud,
        str_to_words=str_to_words,
        words_to_weights=words_to_weights,
        **word_cloud_kwargs,
    )
    value_decoder = Pipe(_word_cloud, wc_decoder)
    return wrap_kvs(text_store, value_decoder=value_decoder)
