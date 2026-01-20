Installation
============

Requirements
------------

- Python >= 3.9
- NumPy >= 1.24.0

Install from PyPI
-----------------

The easiest way to install quickmp is via pip:

.. code-block:: bash

   pip install quickmp

.. note::

   Pre-built wheels from PyPI include the CPU backend only.
   To use the NEC Vector Engine backend, you must install from source.

Install from Source
-------------------

For development, to use the Vector Engine backend, or to get the latest features,
you can install from source:

.. code-block:: bash

   git clone https://github.com/keichi/quickmp.git
   cd quickmp
   pip install -e .

To include test dependencies:

.. code-block:: bash

   pip install -e '.[test]'
