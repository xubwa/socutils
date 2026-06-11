# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'socutils'
copyright = '2024, Xubo Wang and contributors'
author = 'Xubo Wang and contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.mathjax',     # LaTeX math in the theory pages
    'sphinx.ext.napoleon',    # NumPy/Google docstrings (for a future API pass)
    'sphinx.ext.intersphinx', # cross-link to the PySCF / NumPy docs
]

# No autodoc/autosummary API generation in this round: the public API surface
# is curated by hand in the user guides so that internal/scratch methods are
# not exposed.

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pyscf': ('https://pyscf.org', None),
}

templates_path = ['_templates']
exclude_patterns = []
root_doc = 'index'

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'socutils documentation'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}
