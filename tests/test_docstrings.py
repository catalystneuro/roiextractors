import inspect
import os
import importlib
import roiextractors
import pytest


def traverse_class(cls, objs):
    """Traverse a class and its methods and append them to objs."""
    for name, obj in inspect.getmembers(cls, inspect.isfunction or inspect.ismethod):
        objs.append(obj)


def traverse_module(module, objs):
    """Traverse all classes and functions in a module and append them to objs."""
    objs.append(module)
    for name, obj in inspect.getmembers(module, inspect.isclass or inspect.isfunction or inspect.ismethod):
        parent_package = obj.__module__.split(".")[0]
        if parent_package != "roiextractors":  # avoid traversing external dependencies
            continue
        objs.append(obj)
        if inspect.isclass(obj):
            traverse_class(obj, objs)


def traverse_package(package, objs):
    """Traverse all modules and subpackages in a package to append all members to objs."""
    for child in os.listdir(package.__path__[0]):
        if child.startswith(".") or child == "__pycache__":
            continue
        elif child.endswith(".py"):
            module_name = child[:-3]
            module = importlib.import_module(f"{package.__name__}.{module_name}")
            traverse_module(module, objs)
        else:  # subpackage
            subpackage = importlib.import_module(f"{package.__name__}.{child}")
            traverse_package(subpackage, objs)


objs = []
traverse_package(roiextractors, objs)
print(objs)


@pytest.mark.parametrize("obj", objs)
def test_has_docstring(obj):
    """Check if an object has a docstring."""
    doc = inspect.getdoc(obj)
    if inspect.ismodule(obj):
        msg = f"{obj.__name__} has no docstring."
    else:
        msg = f"{obj.__module__}.{obj.__qualname__} has no docstring."
        if "__create_fn__" in msg:
            return  # skip dataclass functions created by __create_fn__
    assert doc is not None, msg


if __name__ == "__main__":
    for obj in objs:
        test_has_docstring(obj)
