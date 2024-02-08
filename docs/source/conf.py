import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "RoiExtractors"
copyright = "2021, Saksham Sharda"
author = "Saksham Sharda"

extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # Includes documentation from docstrings in docs/api
    "sphinx.ext.autosummary",  # Not clear. Please add if you know
    "sphinx.ext.intersphinx",  # Allows links to other sphinx project documentation sites
    "sphinx_search.extension",  # Allows for auto search function the documentation
    "sphinx.ext.viewcode",  # Shows source code in the documentation
    "sphinx.ext.extlinks",  # Allows to use shorter external links defined in the extlinks variable.
]

templates_path = ["_templates"]
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "catalystneuro",
    "github_repo": "roiextractors",
    "github_version": "main",
    "doc_path": "docs",
}

linkcheck_anchors = False

# --------------------------------------------------
# Extension configuration
# --------------------------------------------------

# Napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

# Autodoc
autoclass_content = "both"  # Concatenates docstring of the class with that of its __init__
autodoc_member_order = "bysource"  # Displays classes and methods by their order in source code
autodata_content = "both"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "private-members": True,
    "show-inheritance": False,
    "toctree": True,
}
add_module_names = False

# Intersphinx
intersphinx_mapping = {
    "hdmf": ("https://hdmf.readthedocs.io/en/stable/", None),
    "pynwb": ("https://pynwb.readthedocs.io/en/stable/", None),
    "spikeinterface": ("https://spikeinterface.readthedocs.io/en/latest/", None),
}

# To shorten external links
extlinks = {
    "nwbinspector": ("https://nwbinspector.readthedocs.io/en/dev/%s", ""),
}
