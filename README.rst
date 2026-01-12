********
screen19
********

.. image:: https://img.shields.io/pypi/v/screen19.svg
        :target: https://pypi.python.org/pypi/screen19
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/pyversions/screen19.svg
        :target: https://pypi.org/project/screen19
        :alt: Supported Python versions

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
        :target: https://github.com/astral-sh/ruff
        :alt: Code style: Ruff

Screening program for small-molecule single-crystal X-ray diffraction data

Install as a user
=================
To use screen19, you need to have a Python environment containing DIALS_.
Instructions for installing DIALS can be found here_.
You can then install screen19 with::

    libtbx.pip install screen19

Install as a developer
======================
If you want to develop screen19 then check out a local copy of the repository and install it with::

    git clone git@github.com:xia2/screen19.git
    libtbx.pip install -e screen19

You then need to rebuild libtbx to get the screen19 dispatcher.

.. _DIALS: https://dials.github.io/
.. _here: https://dials.github.io/installation.html
