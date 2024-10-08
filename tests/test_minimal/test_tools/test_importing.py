import pytest
from roiextractors.tools.importing import get_package
import sys
from platform import python_version


def test_get_package_in_sys_modules():
    package_name = "roiextractors"
    package = get_package(package_name=package_name)
    assert package.__name__ == package_name


def test_get_package_not_in_sys_modules():
    package_name = "roiextractors"
    del sys.modules[package_name]
    package = get_package(package_name=package_name)
    assert package.__name__ == package_name


def test_get_package_excluded_versions():
    package_name = "invalid-package"
    excluded_version = python_version()
    excluded_platform = sys.platform
    excluded_platforms_and_python_versions = {excluded_platform: [excluded_version]}
    expected_error = (
        f"\nThe package '{package_name}' is not available on the {excluded_platform} platform for "
        f"Python version {excluded_version}!"
    )
    with pytest.raises(ModuleNotFoundError, match=expected_error):
        package = get_package(
            package_name=package_name, excluded_platforms_and_python_versions=excluded_platforms_and_python_versions
        )


@pytest.mark.parametrize(
    "installation_instructions", [None, "conda install -c conda-forge my-package-name", "pip install my-package-name"]
)
def test_get_package_not_installed(installation_instructions):
    package_name = "invalid-package"
    expected_installation_instructions = installation_instructions or f"pip install {package_name}"
    expected_error = (
        f"\nThe required package'{package_name}' is not installed!\n"
        f"To install this package, please run\n\n\t{expected_installation_instructions}\n"
    )
    with pytest.raises(ModuleNotFoundError, match=expected_error):
        package = get_package(package_name=package_name, installation_instructions=installation_instructions)
