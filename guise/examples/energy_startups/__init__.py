"""Studying energy startups

>>> wc = word_cloud(['here', 'are', 'some', 'words'])
>>> type(wc.to_image())
<class 'PIL.Image.Image'>


"""

import os
from functools import partial
import pandas as pd
import requests
from html2text import html2text
from dol import wrap_kvs
from lexis import Lemmas

lemmas = Lemmas()

myhtml2text = lambda html: ' '.join(
    filter(lemmas.__contains__, html2text(html).split(' '))
)


def get_data():
    from guise.util import proj_files

    data_files = proj_files / 'examples' / 'energy_startups' / 'data'
    df = pd.read_excel(data_files / 'energy5.xlsx')
    df.columns = list(map(str.lower, df.columns))
    return df


def name_url_iter(df):
    for _, row in df.iterrows():
        technology, company, country = map(
            lambda x: row[x].strip(), ('technology', 'company', 'country')
        )
        company = row.company.replace(' ', '_').strip()
        yield f'{technology},{company},{country}', row['website']


def url_to_html(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.content.decode('latin-1')
    except Exception as e:
        print(f'ERROR with {url}: {e}')


DFLT_HTML_STORE_DIR = '~/odat/energy_startups/html'


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


def scrape_and_store(
    name_and_url_iterable, html_store=None, overwrite=False, verbose=True
):
    html_store = get_html_store(html_store)
    for name, url in name_and_url_iterable:
        if overwrite and name in html_store:
            if verbose:
                print(f'Skipping because I already have: {name=}: {url=}')
        else:
            if verbose:
                print(f'Processing {url=}')
            html = url_to_html(url)
            if html is not None:
                html_store[f'{name}.html'] = html





def get_text_store(html_store=None, html2text=myhtml2text):
    _html_2_text_store = wrap_kvs(
        key_of_id=lambda x: x[: -len('.html')],
        id_of_key=lambda x: x + '.html',
        obj_of_data=html2text,
    )
    return _html_2_text_store(get_html_store(html_store))


from flair.models import TextClassifier
from flair.data import Sentence
from functools import lru_cache


DLFT_TEXT_CLASSIFIER_LANG = 'en-sentiment'


@lru_cache
def _get_text_classifier(lang=DLFT_TEXT_CLASSIFIER_LANG):
    return TextClassifier.load(lang)


def _sentiment_score_object(string, lang=DLFT_TEXT_CLASSIFIER_LANG):
    sia = _get_text_classifier(lang)

    sentence = Sentence(string)
    sia.predict(sentence)
    return sentence.labels[0]


def sentiment_score(string):
    score = _sentiment_score_object(string)
    if score.value == 'NEGATIVE':
        return -score.score
    elif score.value == 'POSITIVE':
        return score.score
    else:
        raise ValueError(f"Didn't know score.value could be {score.value}")


def word_cloud(words, save_filepath=None, width=600, height=600, **kwargs):
    from wordcloud import WordCloud
    from collections import Counter

    if isinstance(words, (list, tuple)):
        weight_for_word = Counter(words)
    else:
        weight_for_word = words
    wc = WordCloud(width=width, height=height, **kwargs)
    wc.fit_words(weight_for_word)
    if save_filepath is not None:
        wc.to_file(save_filepath)
    return wc


def _score_to_index(score, max_index, min_score=-1, max_score=1):
    return int(max_index * (score - min_score) / (max_score - min_score))


def _get_red_to_blue_colors(n=202):
    from colour import Color

    red = Color('red')
    colors = list(map(str, red.range_to(Color('blue'), n)))[1:-1]
    return colors


def mk_word_score_base_color_func(word_to_score=sentiment_score, colors=None):
    colors = colors or _get_red_to_blue_colors()

    score_to_index = partial(
        _score_to_index, max_index=len(colors), min_score=-1, max_score=1
    )

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        score = word_to_score(word)
        return colors[score_to_index(score)]

    return color_func
