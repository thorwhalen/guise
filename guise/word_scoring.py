"""Tools to get scores from words"""

from functools import partial

# ---------------------------------------------------------------------------------------
# sentiment scoring

from functools import lru_cache

DLFT_TEXT_CLASSIFIER_LANG = 'en-sentiment'


@lru_cache
def _get_text_classifier(lang=DLFT_TEXT_CLASSIFIER_LANG):
    from flair.models import TextClassifier

    return TextClassifier.load(lang)


def _sentiment_score_object(string, lang=DLFT_TEXT_CLASSIFIER_LANG):
    from flair.data import Sentence

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


# ---------------------------------------------------------------------------------------
# score to color


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
