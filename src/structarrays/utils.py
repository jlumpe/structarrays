"""Misc utility code."""

from collections.abc import Sequence
import numpy as np


class Missing:
	"""Sentinel for a missing value."""

	_instance: 'Missing | None' = None

	def __new__(cls) -> 'Missing':
		# Singleton
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	def __repr__(self) -> str:
		return f'{type(self).__name__}()'


def shape_matches(shape: Sequence[int | None], expected: Sequence[int | None]) -> bool:
	"""Check if the given array shape matches an expected shape.

	Elements of ``expected`` that are ``None`` or negative are ignored.
	"""
	if len(shape) != len(expected):
		return False

	for actual, ex in zip(shape, expected):
		if ex is None or ex < 0:
			continue
		if ex != actual:
			return False

	return True


def format_shape(shape: Sequence[int | None]) -> str:
	"""Format a shape for display."""
	return ', '.join('?' if i is None or i < 0 else str(i) for i in shape)


def check_shape(array: np.ndarray, shape: Sequence[int | None], desc: str = 'array') -> None:
	"""Raise a ``ValueError`` if the array does not have the expected shape."""
	if not shape_matches(array.shape, shape):
		expected_str = format_shape(shape)
		actual_str = format_shape(array.shape)
		raise ValueError(f'{desc} has incorrect shape: expected ({expected_str}), got ({actual_str})')
