# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "quickmp"
copyright = "2025, Keichi Takahashi"
author = "Keichi Takahashi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# -- Autodoc configuration ---------------------------------------------------

autodoc_member_order = "bysource"


def simplify_nanobind_signature(app, what, name, obj, options, signature, return_annotation):
    """Simplify nanobind type annotations for better readability and intersphinx linking."""
    import re

    def simplify(s):
        if s is None:
            return None
        # numpy.ndarray[...] -> numpy.ndarray
        s = re.sub(r"numpy\.ndarray\[[^\]]+\]", "numpy.ndarray", s)
        return s

    return simplify(signature), simplify(return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", simplify_nanobind_signature)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_theme_options = {
    "source_repository": "https://github.com/keichi/quickmp",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
