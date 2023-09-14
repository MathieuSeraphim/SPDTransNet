from importlib import import_module
from importlib.machinery import SourceFileLoader
from os.path import join


def import_class_from_path(class_path: str):
    class_path_as_list = class_path.split(".")
    try:
        class_file_path = join(*class_path_as_list[:-1]) + ".py"
        class_file_module = SourceFileLoader(class_path_as_list[-2], class_file_path).load_module()
    except FileNotFoundError:
        class_file_path = ".".join(class_path_as_list[:-1])
        class_file_module = import_module(class_file_path)
    return class_file_module.__dict__[class_path_as_list[-1]]