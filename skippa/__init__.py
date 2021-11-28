"""Top-level package for skippa.

N.B. Since this imports other modules, setup.py actually requires
Skippa and dependency packages since they are all loaded. Because setup.py
tries to read __version__ from this file.
"""

__author__ = """Robert van Straalen"""
__email__ = 'tech@datasciencelab.nl'
__version__ = '0.1.2'

from .pipeline import Skippa, columns
