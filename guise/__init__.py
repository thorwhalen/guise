"""Semantic fingerprinting

>>> from guise.examples.energy_startups import word_cloud
>>> wc = word_cloud(['here', 'are', 'some', 'words'])
>>> type(wc.to_image())
<class 'PIL.Image.Image'>
"""

from guise.util import proj_files