# guise.util

Utils

### Functions

| `all_links_of_html`(html[, href_value_pattern])                                                    |                                                                               |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`conditional_print`](#guise.util.conditional_print)(\*args, condition, \*\*kwargs)  | Print on `condition`                                                          |
| `get_html_store`([html_store])                                                                     |                                                                               |
| `get_pyversion`()                                                                                  |                                                                               |
| `google_results_urls`(html[, postproc])                                                            |                                                                               |
| [`google_search_html`](#guise.util.google_search_html)(q[, num, start, lr, on_error]) | Get's the html of google results (`start` through `start+num`) for query `q`. |
| `make_dflt_html_store_dir`([store_dir])                                                            |                                                                               |
| `print_status_code_and_content`(response)                                                          |                                                                               |
| `url_to_html`(url[, encoding])                                                                     |                                                                               |
| `without_nones`(d)                                                                                 |                                                                               |

### guise.util.DFLT_STR_TO_WORD(string, pos=0, endpos=9223372036854775807)

Return a list of all non-overlapping matches of pattern in string.

### guise.util.conditional_print(\*args, condition, \*\*kwargs)

Print on `condition`

Intended use:

```pycon
>>> verbose = True  # in the args of a function, and then in the function do:
>>> _print = partial(conditional_print, condition=verbose)
>>> # ... then use _print within function
```

### guise.util.google_search_html(q, num=10, start=0, lr=None, on_error=<function print_status_code_and_content>, \*\*kwargs)

Get’s the html of google results (`start` through `start+num`) for query `q`.

* **Parameters:**
  * **q** – query
  * **num** – the number of results you want (default is 10 – docs say
    it’s also the max, but haven’t seen that to be true)
  * **start** – Which result to start with (play with start and num to page through
    results
  * **lr** – Language restriction (e.g. “lang_en” for English)
  * **on_error** – if given, func that is called on response if status code not 200
  * **kwargs** – Extra params:

See [https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list](https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list)
for extra parameters and descriptions.
