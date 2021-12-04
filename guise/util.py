"""Utils"""


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

