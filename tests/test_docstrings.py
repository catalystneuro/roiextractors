import inspect
from types import ModuleType, FunctionType
from typing import List
import pytest
import roiextractors


def traverse_class(class_object: type, parent: str) -> List[FunctionType]:
    """Traverse the class dictionary and return the methods overridden by this module.

    Parameters
    ----------
    class_object : type
        The class to traverse.
    parent : str
        The parent package name.

    Returns
    -------
    class_methods : List[FunctionType]
        A list of all methods defined in the given class.
    """
    class_methods = []
    for attribute_name, attribute_value in class_object.__dict__.items():
        if isinstance(attribute_value, FunctionType) and attribute_value.__module__.startswith(parent):
            if attribute_name.startswith("__") and attribute_name.endswith("__"):
                continue
            class_methods.append(attribute_value)
    return class_methods


def traverse_module(module: ModuleType, parent: str) -> List:
    """Traverse the module and return all classes and functions defined along the way.

    Parameters
    ----------
    module : ModuleType
        The module to traverse.
    parent : str
        The parent package name.

    Returns
    -------
    local_classes_and_functions : List
        A list of all classes and functions defined in the given module.
    """
    local_classes_and_functions = []

    for name in dir(module):
        if name.startswith("__") and name.endswith("__"):  # skip all magic methods
            continue

        object_ = getattr(module, name)

        if isinstance(object_, type) and object_.__module__.startswith(parent):  # class
            class_object = object_
            class_functions = traverse_class(class_object=class_object, parent=parent)
            local_classes_and_functions.append(class_object)
            local_classes_and_functions.extend(class_functions)

        elif isinstance(object_, FunctionType) and object_.__module__.startswith(parent):
            function = object_
            local_classes_and_functions.append(function)

    return local_classes_and_functions


def traverse_package(package: ModuleType, parent: str) -> List[ModuleType]:
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
            subpackage_members = traverse_package(package=subpackage, parent=parent)
            local_packages_and_modules.append(subpackage)
            local_packages_and_modules.extend(subpackage_members)

        elif isinstance(object_, ModuleType) and object_.__package__.startswith(parent):
            module = object_
            module_members = traverse_module(module=module, parent=parent)
            local_packages_and_modules.append(module)
            local_packages_and_modules.extend(module_members)

    return local_packages_and_modules


objs = traverse_package(roiextractors, "roiextractors")


@pytest.mark.parametrize("obj", objs)
def test_has_docstring(obj):
    """Check if an object has a docstring."""
    doc = inspect.getdoc(obj)
    if inspect.ismodule(obj):
        msg = f"{obj.__name__} has no docstring."
    else:
        msg = f"{obj.__module__}.{obj.__qualname__} has no docstring."
    assert doc is not None, msg


if __name__ == "__main__":
    for obj in objs:
        test_has_docstring(obj)
