import inspect
import os
import importlib
from pathlib import Path
from types import ModuleType, FunctionType
from typing import List, Iterable
from pprint import pprint

import roiextractors


def traverse_class(cls, objs):
    """Traverse a class and its methods and append them to objs."""
    predicate = lambda x: inspect.isfunction(x) or inspect.ismethod(x)
    for name, obj in inspect.getmembers(cls, predicate=predicate):
        objs.append(obj)


def traverse_module(module, objs):
    """Traverse all classes and functions in a module and append them to objs."""
    objs.append(module)
    predicate = lambda x: inspect.isclass(x) or inspect.isfunction(x) or inspect.ismethod(x)
    for name, obj in inspect.getmembers(module, predicate=predicate):
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
        elif Path(child).is_dir():  # subpackage - I did change this one line b/c error otherwise when hit a .json
            subpackage = importlib.import_module(f"{package.__name__}.{child}")
            traverse_package(subpackage, objs)


def traverse_class_2(class_object: type, parent: str) -> List[FunctionType]:
    """Traverse the class dictionary and return the methods overridden by this module."""
    class_functions = list()
    for attribute_name, attribute_value in class_object.__dict__.items():
        if isinstance(attribute_value, FunctionType) and attribute_value.__module__.startswith(parent):
            class_functions.append(attribute_value)
    return class_functions


def traverse_module_2(module: ModuleType, parent: str) -> Iterable[FunctionType]:
    """Traverse the module directory and return all submodules, classes, and functions defined along the way."""
    local_modules_classes_and_functions = list()

    for name in dir(module):
        if name.startswith("__") and name.endswith("__"):  # skip all magic methods
            continue

        object_ = getattr(module, name)

        if isinstance(object_, ModuleType) and object_.__package__.startswith(parent):
            submodule = object_

            submodule_functions = traverse_module_2(module=submodule, parent=parent)

            local_modules_classes_and_functions.append(submodule)
            local_modules_classes_and_functions.extend(submodule_functions)
        elif isinstance(object_, type) and object_.__module__.startswith(parent):  # class
            class_object = object_

            class_functions = traverse_class_2(class_object=class_object, parent=parent)

            local_modules_classes_and_functions.append(class_object)
            local_modules_classes_and_functions.extend(class_functions)
        elif isinstance(object_, FunctionType) and object_.__module__.startswith(parent):
            function = object_

            local_modules_classes_and_functions.append(function)

    return local_modules_classes_and_functions


def traverse_class3(class_object: type, parent: str) -> List[FunctionType]:
    """Traverse the class dictionary and return the methods overridden by this module."""
    class_functions = []
    for attribute_name, attribute_value in class_object.__dict__.items():
        if isinstance(attribute_value, FunctionType) and attribute_value.__module__.startswith(parent):
            if attribute_name.startswith("__") and attribute_name.endswith("__"):
                continue
            class_functions.append(attribute_value)
    return class_functions


def traverse_module3(module: ModuleType, parent: str) -> List:
    local_classes_and_functions = []

    for name in dir(module):
        if name.startswith("__") and name.endswith("__"):  # skip all magic methods
            continue

        object_ = getattr(module, name)

        if isinstance(object_, type) and object_.__module__.startswith(parent):  # class
            class_object = object_
            class_functions = traverse_class3(class_object=class_object, parent=parent)
            local_classes_and_functions.append(class_object)
            local_classes_and_functions.extend(class_functions)

        elif isinstance(object_, FunctionType) and object_.__module__.startswith(parent):
            function = object_
            local_classes_and_functions.append(function)

    return local_classes_and_functions


def traverse_package3(package: ModuleType, parent: str) -> List[ModuleType]:
    """Traverse the package and return all subpackages and modules defined along the way.

    Parameters
    ----------
    package : ModuleType
        The package, subpackage, or module to traverse.
    parent : str
        The parent package name.

    Returns
    -------
    local_packages_and_modules : List[ModuleType]
        A list of all subpackages and modules defined in the given package.
    """
    local_packages_and_modules = []

    for name in dir(package):
        if name.startswith("__") and name.endswith("__"):  # skip all magic methods
            continue

        object_ = getattr(package, name)

        if (
            isinstance(object_, ModuleType)
            and object_.__file__[-11:] == "__init__.py"
            and object_.__package__.startswith(parent)
        ):
            subpackage = object_
            subpackage_members = traverse_package3(package=subpackage, parent=parent)
            local_packages_and_modules.append(subpackage)
            local_packages_and_modules.extend(subpackage_members)

        elif isinstance(object_, ModuleType) and object_.__package__.startswith(parent):
            module = object_
            module_members = traverse_module3(module=module, parent=parent)
            local_packages_and_modules.append(module)
            local_packages_and_modules.extend(module_members)

    return local_packages_and_modules


list_1 = list()
traverse_package(package=roiextractors, objs=list_1)

list_2 = traverse_module_2(module=roiextractors, parent="roiextractors")
list_3 = traverse_package3(package=roiextractors, parent="roiextractors")

# Analyze and compare - note that for set comparison, the lists must have been run in the same kernel
# to give all imports the same address in memory
unique_list_1 = set(list_1)
unique_list_2 = set(list_2)
unique_list_3 = set(list_3)

found_by_2_and_not_by_1 = unique_list_2 - unique_list_1
print("found by 2 and not by 1:")
pprint(found_by_2_and_not_by_1)

# Summary: A series of nested submodules under `checks` and `tools`; some various private functions scattered around
# not really clear why Paul's missed these

found_by_1_and_not_by_2 = unique_list_1 - unique_list_2
print("found by 1 and not by 2:")
pprint(found_by_1_and_not_by_2)

# Summary: All of these are bound methods of the Enum's (Importance/Severity) or JSONEncoder
# and are not methods that we actually override in the codebase (they strictly inherit)
# It did, however, find the outermost package __init__ (does that really need a docstring though?)

found_by_3_and_not_by_2 = unique_list_3 - unique_list_2
print("found by 3 and not by 2:")
pprint(found_by_3_and_not_by_2)

found_by_2_and_not_by_3 = unique_list_2 - unique_list_3
print("found by 2 and not by 3:")
pprint(found_by_2_and_not_by_3)
