# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

for x in os.walk('../hypergal'):
  sys.path.insert(0, x[0])

from hypergal import *
from hypergal import fit


# -- Project information -----------------------------------------------------

project = 'hypergal'
copyright = '2021, J.Lezmy, M.Rigault'
author = 'J.Lezmy, M.Rigault'

# The full version, including alpha/beta/rc tags
release = '1.0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [# Standard extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',       # or pngmath
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx.ext.extlinks',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.coverage',
    # Other extensions  
]

#autoapi_dirs = ['../hypergal']

#autoapi_options = ['members', 'undoc-members', 'private-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members','inherited-members','show-inheritance-diagram']


inheritance_node_attrs = dict(shape='ellipse', fontsize=13, height=0.75,
                              color='sienna', style='filled', imagepos='tc')

inheritance_graph_attrs = dict(rankdir="LR", size='""')

autoclass_content = "both"              # Insert class and __init__ docstrings
autodoc_member_order = "bysource"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas-docs.github.io/pandas-docs-travis/', None),
    'iminuit': ('https://iminuit.readthedocs.io/en/latest/', None),
    #'emcee': ('https://emcee.readthedocs.io/en/latest', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
#master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build','build', 'Thumbs.db', '.DS_Store','requirements.txt', 'hypergal.egg-info',
                    ]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
