# guise.nlp

Natural language processing tools.

Also check out the package idiom and lexis for more tools.
(A few functions in this module use idiom and lexis, dynamically imported.)

### Functions

| `english_word_frequencies`([max_num_of_words])                                                   |                                                                                |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`ensure_corpus_counts`](#guise.nlp.ensure_corpus_counts)(word_count_src, \*[, ...]) | Helper function to get a word                                                  |
| [`get_include_terms`](#guise.nlp.get_include_terms)([include])                    | allowed_terms = 'most_frequent_english' or 'wordnet_lemmas'                    |
| `html_tokens`(html[, include, exclude])                                                          |                                                                                |
| [`map_words_of_word_count`](#guise.nlp.map_words_of_word_count)(word_count, word_map)   | Recompute a word count based on a (complete) mapping of it's words (keys)      |
| [`stem_based_word_count`](#guise.nlp.stem_based_word_count)(words[, stemmer])         | From an iterable of words, get a WordCount mapping which uses a stem-based map |
| [`stem_based_word_mapping`](#guise.nlp.stem_based_word_mapping)(words[, stemmer])       | Returns a dict of {word: replace_with_word, ...} mapping based on stemming.    |
| [`term_filtered_html2text`](#guise.nlp.term_filtered_html2text)(html[, include, ...])   | allowed_terms = 'most_frequent_english' or 'wordnet_lemmas'                    |

### Classes

| [`TFIDFCalculator`](#guise.nlp.TFIDFCalculator)(reference_corpus[, ...])   | A TF-IDF calculator that computes TF-IDF scores for documents based on a corpus.   |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|

### guise.nlp.DFLT_STR_TO_WORD(string, pos=0, endpos=9223372036854775807)

Return a list of all non-overlapping matches of pattern in string.

### *class* guise.nlp.TFIDFCalculator(reference_corpus, idf_type='frequency', stopwords=None, custom_idf=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A TF-IDF calculator that computes TF-IDF scores for documents based on a corpus.

* **Parameters:**
  * **reference_corpus** (*pandas.Series*) – The corpus word counts to use as reference for IDF calculations.
    The `corpus_counts` will be taken, or computed from reference_corpus.
  * **idf_type** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* *optional*) – 

    The type of IDF calculation to use. Options are as follows (where df is the
    ‘document frequency’ of a word (a word count) and N is the total number of
    documents in the corpus):
    - ’standard’: IDF = log(N / (df))
    - ’smooth’: IDF = log(1 + N / (df))
    - ’probabilistic’: IDF = log((N - df) / df)
    - ’frequency’: IDF = log((total_counts_corpus + 1) / (corpus_counts + 1))
      Default is ‘frequency’.
  * **stopwords** ([*set*](https://docs.python.org/3/builtins/stdtypes.html#set) *or* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *,* *optional*) – A set or list of stop words to exclude from calculations.
  * **custom_idf** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *or* *pandas.Series* *,* *optional*) – A mapping from words to custom IDF values.

#### idf_

Inverse Document Frequency (IDF) values computed from the corpus.

* **Type:**
  pandas.Series

#### default_idf_

Default IDF value for words not found in the corpus.

* **Type:**
  [*float*](https://docs.python.org/3/builtins/functions.html#float)

### Examples

```pycon
>>> import pandas as pd
>>> # Sample corpus word counts
>>> reference_corpus = pd.Series({
...     'word1': 300,
...     'word2': 200,
...     'word4': 500,
...     'trump': 100
... })
>>> stopwords = {'the', 'and', 'is'}
>>> custom_idf = {'trump': 5.0}
>>> tfidf_calculator = TFIDFCalculator(reference_corpus,
...                                    idf_type='standard',
...                                    stopwords=stopwords,
...                                    custom_idf=custom_idf)
>>> # Sample document word counts
>>> doc_counts = pd.Series({
...     'word1': 3,
...     'word2': 2,
...     'trump': 4,
...     'harris': 2,
...     'the': 10
... })
>>> tfidf_scores = tfidf_calculator(doc_counts)
>>> print(tfidf_scores)
word1    -1.17...
word2    -0.71...
trump     1.81...
harris    0.90...
dtype: float64
```

### guise.nlp.ensure_corpus_counts(word_count_src, \*, str_to_words=<built-in method findall of re.Pattern object>, words_to_weights=<class 'collections.Counter'>)

Helper function to get a word

* **Return type:**
  `Series`

### guise.nlp.get_include_terms(include='most_frequent_english')

allowed_terms = ‘most_frequent_english’ or ‘wordnet_lemmas’

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### guise.nlp.map_words_of_word_count(word_count, word_map)

Recompute a word count based on a (complete) mapping of it’s words (keys)

```pycon
>>> assert (dict(
... map_words_of_word_count(
...     {'one': 1, 'ones': 2, '1': 3, 'two': 4},
...     {'one': 'one', 'ones': 'one', '1': 'one', 'two': 'two'}))
... == {'one': 6, 'two': 4}
... )
```

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### guise.nlp.stem_based_word_count(words, stemmer='lancaster')

From an iterable of words, get a WordCount mapping which uses a stem-based map

```pycon
>>> words = ['happy', 'happier', 'happiest', 'happily', 'sad', 'sadly']
>>> assert (dict(stem_based_word_count(words))
... == {'happy': 3, 'happiest': 1, 'sad': 2})
```

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### guise.nlp.stem_based_word_mapping(words, stemmer='lancaster')

Returns a dict of {word: replace_with_word, …} mapping based on stemming.

stemming is great to reduce the number of words by replacing several

The words (or word counts) are stemmed, then for each stem, the word with the
highest count gets to represent the others.

```pycon
>>> _words = ['happy', 'happier', 'happiest', 'happily', 'sad', 'sadly']
>>> word_count = Counter(_words)
>>> stem_based_word_mapping(word_count)
{'happy': 'happy',
 'happier': 'happy',
 'happiest': 'happiest',
 'happily': 'happy',
 'sad': 'sad',
 'sadly': 'sad'}
```

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### guise.nlp.term_filtered_html2text(html, include='most_frequent_english', exclude=frozenset({'', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever', 'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hence', 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'therefore', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}), sep=' ; ')

allowed_terms = ‘most_frequent_english’ or ‘wordnet_lemmas’

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
