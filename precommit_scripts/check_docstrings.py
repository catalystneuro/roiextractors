import inspect
import os
import importlib
import roiextractors


def check_docstring(obj):
    """Check if an object has a docstring."""
    doc = inspect.getdoc(obj)
    if doc is None:
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
            print(f"{obj.__module__}.{obj.__name__} has no docstring.")
        else:
            print(f"{obj.__name__} has no docstring.")


def traverse_class(cls):
    """Traverse a class and its methods."""
    for name, obj in inspect.getmembers(cls, inspect.isfunction or inspect.ismethod):
        check_docstring(obj)


def traverse_module(module):
    """Traverse all classes and functions in a module."""
    check_docstring(module)
    for name, obj in inspect.getmembers(module, inspect.isclass or inspect.isfunction or inspect.ismethod):
        parent_package = obj.__module__.split(".")[0]
        if parent_package != "roiextractors":  # avoid traversing external dependencies
            continue
        check_docstring(obj)
        if inspect.isclass(obj):
            traverse_class(obj)


def traverse_package(package):
    """Traverse all modules and subpackages in a package."""
    for child in os.listdir(package.__path__[0]):
        if child.startswith(".") or child == "__pycache__":
            continue
        elif child.endswith(".py"):
            module_name = child[:-3]
            module = importlib.import_module(f"{package.__name__}.{module_name}")
            traverse_module(module)
        else:  # subpackage
            subpackage = importlib.import_module(f"{package.__name__}.{child}")
            traverse_package(subpackage)


if __name__ == "__main__":
    traverse_package(roiextractors)
