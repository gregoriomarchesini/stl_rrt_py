# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'stl_tool'
copyright = '2025, Gregorio Marchesini'
author = 'Gregorio Marchesini'
release = '1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Auto-generates API docs
    "sphinx.ext.autosummary", # Generates summaries of modules/classes
    'sphinx.ext.napoleon',   # Supports Google-style docstrings
    'sphinx.ext.viewcode',   # Adds links to source code
    'sphinx.ext.mathjax',
    'myst_parser',           # Enables Markdown (.md) support
    'nbsphinx'               # Enables Jupyter notebooks rendering
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "module-first": True,  # Show the module name at the top
}

source_suffix = ['.rst', '.md']

# Avoid showing module names in front of function definitions
add_module_names = False 


nbsphinx_execute = 'never'  # Avoids executing notebooks during doc build


templates_path = ['_templates']
exclude_patterns = ['../build', '../tests', '.../openmpc.egg-info', '../OpenMPC.egg-info','_build']

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# logo
# html_logo = '_static/logo.png'
html_theme_options = {
    'logo_only': True,
}

html_sidebars = {
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
}