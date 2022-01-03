"""Semantic fingerprinting

>>> from guise import word_cloud
>>> wc = word_cloud(['here', 'are', 'some', 'words'])
>>> type(wc.to_image())
<class 'PIL.Image.Image'>
"""

from guise.util import (
    proj_files,
    google_search_html,
    google_results_urls,
    url_to_html,
    all_links_of_html,
)
from guise.nlp import (
    html_tokens,
    term_filtered_html2text,
    stem_based_word_mapping,
    stem_based_word_count,
    DFLT_INCLUDE_TERMS,
    DFLT_EXCLUDE_TERMS,
)
from guise.word_scoring import (
    sentiment_score,
    mk_word_score_base_color_func,
)
from guise.tools import google_results_toks, word_cloud
