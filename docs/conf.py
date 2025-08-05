import os, sys
sys.path.insert(0, os.path.abspath('..'))
project = 'explainboostedregg'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon']
html_theme = 'alabaster'
