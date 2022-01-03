"""Tools"""
from typing import Mapping
from functools import partial
from itertools import chain
from collections import Counter
from inspect import signature

from wordcloud import WordCloud

from guise.util import google_search_html, google_results_urls, url_to_html
from guise.nlp import html_tokens, DFLT_INCLUDE_TERMS, DFLT_EXCLUDE_TERMS

DFLT_URL_TO_HTML_KWARGS = (('timeout', 20),)

# TODO: Can be accelerated significantly by async
def google_results_toks(
    q,
    *,
    num=30,
    lr='lang_en',
    include=DFLT_INCLUDE_TERMS,
    exclude=DFLT_EXCLUDE_TERMS,
    verbose=True,
    url_to_html_kwargs: dict = DFLT_URL_TO_HTML_KWARGS,
):
    """A generator of tokens """
    # get the google results html for query q
    results_html = google_search_html(q, num=num, lr=lr)

    # parse out the results urls
    results_urls = google_results_urls(results_html)

    # get the landing page (html) of each one of those result urls
    htmls = [url_to_html(url, **dict(url_to_html_kwargs)) for url in results_urls]

    # make a list of urls whose htmls couldn't be acquired
    # (status_code>200, or other kind of problem)
    problematic_urls = [results_urls[i] for i, html in enumerate(htmls) if html is None]

    # filter out the "bad" results
    htmls = list(filter(None, htmls))
    if problematic_urls and verbose:
        print('There were some problematic urls:')
        print(*problematic_urls, sep='\n')

    # return an iterator of tokens (words/terms) extracted from these htmls
    tokenizer = partial(html_tokens, include=include, exclude=exclude)
    return chain.from_iterable(map(tokenizer, htmls))


def word_cloud(
    words,
    save_filepath=None,
    *,
    color_func=None,
    stopwords=None,
    width=600,
    height=600,
    random_state=42,  # set to None to get a different image every time
    font_path=None,
    margin=2,
    ranks_only=None,
    prefer_horizontal=0.9,
    mask=None,
    scale=1,
    max_words=200,
    min_font_size=4,
    background_color='black',
    max_font_size=None,
    font_step=1,
    mode='RGB',
    relative_scaling='auto',
    regexp=None,
    collocations=True,
    colormap=None,
    normalize_plurals=True,
    contour_width=0,
    contour_color='black',
    repeat=False,
    include_numbers=False,
    min_word_length=0,
    collocation_threshold=30,
):
    word_cloud_kwargs = {
        k: v for k, v in locals().items() if k in set(signature(WordCloud).parameters)
    }
    if isinstance(words, str):
        # simple tokenization of string into words
        words = list(map(str.strip, words.split(' ')))
    if isinstance(words, Mapping):
        # We already have a {word: frequency, ...} mapping
        weight_for_word = words
    else:
        # assume we're given an iterable of words and make a {word: count,...) dict
        weight_for_word = Counter(words)

    wc = WordCloud(**word_cloud_kwargs)
    wc.fit_words(weight_for_word)
    if save_filepath is not None:
        wc.to_file(save_filepath)
    return wc
