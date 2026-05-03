import numpy as np

from structarrays import StructArray, field


# ------------------------------------------------------------------------------------------------ #
#                                               Utils                                              #
# ------------------------------------------------------------------------------------------------ #

def array_scalar_equal(value1, value2) -> bool:
	"""Whether two arrays or scalars are equal.

	Does not consider 0d arrays equal to scalars.
	"""
	return (
		np.isscalar(value1) == np.isscalar(value2)
		and np.array_equal(value1, value2)
	)


def same_memory(arr1: StructArray | np.ndarray, arr2: StructArray | np.ndarray):
	"""Whether the two (struct) arrays share exactly the same memory."""
	if isinstance(arr1, StructArray):
		arr1 = arr1.array
	if isinstance(arr2, StructArray):
		arr2 = arr2.array

	if arr1.shape != arr2.shape:
		return False

	# pointer, readonly flag
	p1, _ = arr1.__array_interface__['data']
	p2, _ = arr2.__array_interface__['data']
	return p1 == p2


def disjoint_memory(arr1: StructArray | np.ndarray, arr2: StructArray | np.ndarray):
	"""Whether the two (struct) arrays do not share any memory."""
	if isinstance(arr1, StructArray):
		arr1 = arr1.array
	if isinstance(arr2, StructArray):
		arr2 = arr2.array
	return not np.shares_memory(arr1, arr2)


# ------------------------------------------------------------------------------------------------ #
#                                             Examples                                             #
# ------------------------------------------------------------------------------------------------ #

class SimpleStruct(StructArray):
	"""Struct with scalar, vector, and matrix fields."""

	x = field(None, default=1.5)
	y = field(3)
	z = field((2, 3), default=[[11, 12, 13], [21, 22, 23]])


class NestedInner(StructArray):
	"""Inner struct for nesting tests."""

	a = field(None, default=10.0)
	b = field(2)


class NestedStruct(StructArray):
	"""Struct with nested StructArray field."""

	inner = field(NestedInner)
	z = field(None, default=3.0)


class StructWithDefaults(StructArray):
	"""Struct with default_factory."""

	vec = field(3, default_factory=lambda: np.array([1, 2, 3]))
