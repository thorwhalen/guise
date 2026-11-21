# guise.word_clouds

Word cloud generation with wordcloud.WordCloud.

### Functions

| `vectorize_word_weighting_func`(...)                                                             |                                              |
|--------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`word_cloud`](#guise.word_clouds.word_cloud)([words, save_filepath, ...])         | Create a word cloud from a list of words.    |
| [`word_cloud_store`](#guise.word_clouds.word_cloud_store)(text_store[, wc_decoder, ...]) | Create a word cloud store from a text store. |
| `words_to_word_weights_dict`(words, ...)                                                         |                                              |

### guise.word_clouds.DFLT_STR_TO_WORD(string, pos=0, endpos=9223372036854775807)

Return a list of all non-overlapping matches of pattern in string.

### guise.word_clouds.word_cloud(words=None, save_filepath=None, \*, str_to_words=<built-in method findall of re.Pattern object>, words_to_weights=<class 'collections.Counter'>, wc_decoder=<function <lambda>>, stopwords=None, font_path=None, width=400, height=200, margin=2, ranks_only=None, prefer_horizontal=0.9, mask=None, scale=1, color_func=None, max_words=200, min_font_size=4, random_state=None, background_color='black', max_font_size=None, font_step=1, mode='RGB', relative_scaling='auto', regexp=None, collocations=True, colormap=None, normalize_plurals=True, contour_width=0, contour_color='black', repeat=False, include_numbers=False, min_word_length=0, collocation_threshold=30)

Create a word cloud from a list of words.

The words can be a list of strings or a mapping from words to weights.

The word cloud is created by applying the word_cloud function to the words.

The word_cloud_kwargs are passed to the WordCloud constructor.

### Args

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

### guise.word_clouds.word_cloud_store(text_store, wc_decoder=<function <lambda>>, \*, str_to_words=<built-in method findall of re.Pattern object>, words_to_weights=<class 'collections.Counter'>, stopwords=None, font_path=None, width=400, height=200, margin=2, ranks_only=None, prefer_horizontal=0.9, mask=None, scale=1, color_func=None, max_words=200, min_font_size=4, random_state=None, background_color='black', max_font_size=None, font_step=1, mode='RGB', relative_scaling='auto', regexp=None, collocations=True, colormap=None, normalize_plurals=True, contour_width=0, contour_color='black', repeat=False, include_numbers=False, min_word_length=0, collocation_threshold=30)

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
