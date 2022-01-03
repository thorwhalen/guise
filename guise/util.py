"""Utils"""
import os
import re
from functools import partial, lru_cache

from urllib.parse import quote as url_quote
import requests


def get_pyversion():
    import sys

    _maj, _minor, *_ = sys.version_info
    return _maj, _minor


py_version = get_pyversion()

if py_version >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

proj_name, *_ = __name__.split('.')

proj_files = files(proj_name)
data_files = proj_files / 'data'

STOPWORDS = frozenset(
    map(str.strip, (data_files / 'stopwords.txt').read_text().split('\n'))
)


def conditional_print(*args, condition, **kwargs):
    """Print on `condition`

    Intended use:

    >>> verbose = True  # in the args of a function, and then in the function do:
    >>> _print = partial(conditional_print, condition=verbose)
    >>> # ... then use _print within function
    """
    if condition:
        print(*args, **kwargs)


def url_to_html(url, encoding='latin-1', **kwargs):
    try:
        r = requests.get(url, **kwargs)
        if r.status_code == 200:
            return r.content.decode(encoding=encoding)
    except Exception as e:
        print(f'ERROR with {url}: {e}')



def without_nones(d):
    return {k: v for k, v in d.items() if v is not None}


def print_status_code_and_content(response):
    print(f'ERROR: {response.status_code}: {response.content}')


def google_search_html(
    q, num=10, start=0, lr=None, on_error=print_status_code_and_content, **kwargs
):
    """
    Get's the html of google results (`start` through `start+num`) for query `q`.

    :param q: query
    :param num: the number of results you want (default is 10 -- docs say
        it's also the max, but haven't seen that to be true)
    :param start: Which result to start with (play with start and num to page through
        results
    :param lr: Language restriction (e.g. "lang_en" for English)
    :param on_error: if given, func that is called on response if status code not 200
    :param kwargs: Extra params:

    See https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    for extra parameters and descriptions.


    """
    google_search_url = 'https://www.google.com/search'

    params = without_nones(dict(q=q, start=start, num=num, lr=lr, **kwargs))

    r = requests.get(google_search_url, params=params)
    if r.status_code:
        return r.content
    else:
        if on_error:
            return on_error(r)


def all_links_of_html(html, href_value_pattern='http.*'):
    from bs4 import BeautifulSoup
    href_value_pattern = re.compile(href_value_pattern)
    t = BeautifulSoup(html, features='lxml')
    return list(
        dict.fromkeys(  # this removes duplicates conserving order
            x.get('href')
            for x in t.find_all(attrs={'href': href_value_pattern}, recursive=True)
        )
    )


_google_url_prefix = '/url?q='
_google_url_junk = re.compile(r'&sa=\w+&ved=[\w-]+&usg=[\w-]+$')


def _google_url_post_proc(url):
    if url.startswith(_google_url_prefix):
        url = url[len(_google_url_prefix) :]
    if _google_url_junk.search(url):
        url = _google_url_junk.sub('', url)
    return url


def google_results_urls(html, postproc=_google_url_post_proc):
    return list(
        dict.fromkeys(  # to remove duplicates but keep order
            map(
                postproc,
                filter(lambda url: 'google.com' not in url, all_links_of_html(html)),
            )
        )
    )


DFLT_HTML_STORE_DIR = '~/odat/html'


def make_dflt_html_store_dir(store_dir=DFLT_HTML_STORE_DIR):
    store_dir = os.path.expanduser(store_dir)
    if not os.path.isdir(store_dir):
        answer = input(
            f'Can I make the directory to store htmls in them (Y/n): {store_dir}'
        )
        if answer.lower() in {'y', ''}:
            os.makedirs(store_dir)


def get_html_store(html_store=None):
    if html_store is None:
        html_store = os.path.expanduser(DFLT_HTML_STORE_DIR)
    if isinstance(html_store, str):
        make_dflt_html_store_dir(html_store)
        from py2store import LocalTextStore

        rootdir = html_store
        html_store = LocalTextStore(html_store, rootdir)
    return html_store

