__author__ = 'shiry'
import importlib


def load_dictionary(filename_no_extension):
    """
    :type filename_no_extension: A filename string without the .py extension
    """
    module = importlib.import_module(filename_no_extension)
    assert isinstance(module, object)
    assert isinstance(module.content, dict)
    return module.content


def save_dictionary(dictionary, filename):
    f = open(filename, "w")
    f.write("content = " + str(dictionary))
    f.close()