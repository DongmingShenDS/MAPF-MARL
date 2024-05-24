# DS DONE 1
from collections import namedtuple


def convert(dictionary):
    """convert dictionary to namedtuple"""
    # DS: namedtuple: container similar to dict but have better accessing properties (by name/index)
    # DS: https://docs.python.org/3/library/collections.html
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
