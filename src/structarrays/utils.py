"""Misc utility code."""

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


def check_shape(array: np.ndarray, shape: tuple[int | None, ...], desc: str = 'array') -> None:
	if array.ndim == len(shape):
		for expected, actual in zip(shape, array.shape):
			if expected is not None and expected != actual:
				break
		else:
			return

	expected_str = ', '.join(str(i or '?') for i in shape)
	actual_str = ', '.join(str(i) for i in array.shape)
	raise ValueError(f'{desc} has incorrect shape: expected ({expected_str}), got ({actual_str})')
