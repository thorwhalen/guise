"""Tools"""

from functools import partial
from itertools import chain
from inspect import signature

from guise.util import google_search_html, google_results_urls, url_to_html
from guise.nlp import html_tokens, DFLT_INCLUDE_TERMS, DFLT_EXCLUDE_TERMS
from guise.word_clouds import word_cloud  # for backwards compatibility, importing here


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
    """A generator of tokens"""
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
