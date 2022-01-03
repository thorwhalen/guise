"""Studying energy startups

>>> wc = word_cloud(['here', 'are', 'some', 'words'])
>>> type(wc.to_image())
<class 'PIL.Image.Image'>


"""

from functools import partial
import pandas as pd
from dol import wrap_kvs


from guise.util import (
    url_to_html,
    get_html_store,
    word_cloud,
)

from guise.word_scoring import (
    mk_word_score_base_color_func,
    sentiment_score,
    term_filtered_html2text,
)

EX_DFLT_HTML_STORE_DIR = '~/odat/energy_startups/html'
url_to_html = partial(url_to_html, EX_DFLT_HTML_STORE_DIR)
get_html_store = get_html_store(get_html_store, EX_DFLT_HTML_STORE_DIR)


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


def get_text_store(html_store=None, html2text=term_filtered_html2text):
    _html_2_text_store = wrap_kvs(
        key_of_id=lambda x: x[: -len('.html')],
        id_of_key=lambda x: x + '.html',
        obj_of_data=html2text,
    )
    return _html_2_text_store(get_html_store(html_store))
