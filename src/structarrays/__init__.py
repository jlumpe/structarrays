"""
Structured views of NumPy arrays.
"""

__author__ = 'Jared Lumpe'
__email__ = 'jared@jaredlumpe.com'
__version__ = '0.1.0'


__all__ = [
	'StructArray',
	'Field',
	'arrayfield',
	'ScalarField',
	'ArrayField',
	'StructField',
	'CustomField',
]


from .base import StructArray, Field
from .fields import arrayfield, ScalarField, ArrayField, StructField, CustomField
