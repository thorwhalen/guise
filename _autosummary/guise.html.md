# guise

Semantic fingerprinting

```pycon
>>> from guise import word_cloud
>>> wc = word_cloud(['here', 'are', 'some', 'words'])
>>> type(wc.to_image())
<class 'PIL.Image.Image'>
```

### Modules

| [`nlp`](guise.nlp.html.md#module-guise.nlp)                   | Natural language processing tools.              |
|-----------------------------------------------------------------------------------------|-------------------------------------------------|
| [`tools`](guise.tools.html.md#module-guise.tools)               | Tools                                           |
| [`util`](guise.util.html.md#module-guise.util)                 | Utils                                           |
| [`word_clouds`](guise.word_clouds.html.md#module-guise.word_clouds)   | Word cloud generation with wordcloud.WordCloud. |
| [`word_scoring`](guise.word_scoring.html.md#module-guise.word_scoring) | Tools to get scores from words                  |
