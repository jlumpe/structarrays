"""Test StructArray core functionality."""

import numpy as np
import pytest

from structarrays import StructArray, arrayfield
from structarrays.base import StructArrayFields
from .common import (
	SimpleStruct, NestedStruct, StructWithDefaults, array_scalar_equal, same_memory,
)


class TestInitialization:
	"""Test constructor / initialization."""

	def test_init_default(self):
		"""Test initialization with default (no array) allocates a new array and applies defaults."""
		struct = SimpleStruct()
		assert struct.array.shape == (10,)  # 1 + 3 + 6
		assert struct.x == 1.5
		assert np.array_equal(struct.y, [0, 0, 0])
		assert np.array_equal(struct.z, [[11, 12, 13], [21, 22, 23]])

	def test_init_with_array(self):
		"""Test initialization with provided array wraps it."""
		values = np.arange(10).astype(float)
		arr = values.copy()
		struct = SimpleStruct(arr)
		assert struct.array is arr
		# Not modified
		assert np.array_equal(arr, values)
		assert struct.x == 0.0
		assert np.array_equal(struct.y, [1, 2, 3])
		assert np.array_equal(struct.z, [[4, 5, 6], [7, 8, 9]])

	def test_init_wrong_shape(self):
		"""Test initialization from an array of incorrect shape raises ValueError."""
		# Wrong length
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct(np.zeros(SimpleStruct.size + 1))

		# Empty array
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct(np.zeros(0))

		# Wrong number of dimensions
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct(np.zeros((SimpleStruct.size, 1)))
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct(np.zeros((2, 5)))  # Right total size, wrong shape

		# Array-like with wrong shape gets converted via np.asarray and then checked
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct([1, 2, 3])

	def test_init_invalid_kwarg(self):
		"""Passing a non-field kwarg to ``__init__()`` raises ValueError."""

		with pytest.raises(ValueError, match="Invalid field name"):
			SimpleStruct(nonexistent=1)  # type: ignore

		# Unknown kwarg is rejected even when an array is provided
		with pytest.raises(ValueError, match="Invalid field name"):
			SimpleStruct(np.zeros(SimpleStruct.size), nonexistent=1)  # type: ignore

	def test_default_factory_applied_on_init(self):
		s = StructWithDefaults()
		assert np.array_equal(s.vec, [1, 2, 3])

	def test_default_factory_called_per_instance(self):
		s1 = StructWithDefaults()
		s2 = StructWithDefaults()
		# Each gets its own array from the factory
		assert not np.shares_memory(s1.vec, s2.vec)


class TestCreation:
	"""Test subclass creation."""

	def test_class_metadata(self):
		"""Test fields and  size are set correctly."""
		# size
		assert SimpleStruct.size == 10
		assert SimpleStruct.fields.size == SimpleStruct.size

		# fields
		assert isinstance(SimpleStruct.fields, StructArrayFields)
		assert SimpleStruct.fields.names() == ['x', 'y', 'z']

		# Matches class attributes
		for field in SimpleStruct.fields:
			assert getattr(SimpleStruct, field.name) is field

	def test_field_name_collision(self):
		"""Field name that shadows a StructArray method or class variable raises ValueError."""

		# Not all-inclusive
		for name in ['fields', 'size', 'array', 'copy', 'asdict']:
			with pytest.raises(ValueError, match="Invalid field name"):
				attrs = {
					name: arrayfield(1)
				}
				# Dynamic class creation with field of given name
				type('Bad', (StructArray,), attrs)


class TestFields:
	"""Test field access and assignment."""

	def test_assignment(self):
		"""Test setting field values."""

		struct = SimpleStruct()

		xval = 10
		struct.x = xval
		assert struct.x == xval

		yval = [11, 12, 13]
		struct.y = yval  # type: ignore
		assert np.array_equal(struct.y, yval)

		zval = [[14, 15, 16], [17, 18, 19]]
		struct.z = zval  # type: ignore
		assert np.array_equal(struct.z, zval)

		assert np.array_equal(struct.array, np.arange(10, 20))


class TestMethods:
	"""Test misc methods."""

	def test_asdict(self):
		"""Test asdict() method with raw=False."""
		struct = SimpleStruct(np.arange(10, 20))
		d = struct.asdict()
		assert d.keys() == SimpleStruct.fields.by_name.keys()
		for k, v in d.items():
			assert array_scalar_equal(v, getattr(struct, k))

	def test_asdict_raw(self):
		"""Test asdict() method with raw=True."""
		struct = SimpleStruct(np.arange(10, 20))
		arrs = struct.asdict(raw=True)
		assert arrs.keys() == SimpleStruct.fields.by_name.keys()
		for k, v in arrs.items():
			field = getattr(SimpleStruct, k)
			assert same_memory(v, field.get_raw(struct))

	def test_copy(self):
		struct1 = SimpleStruct(np.arange(10))
		struct2 = struct1.copy()
		assert isinstance(struct2, SimpleStruct)
		assert np.array_equal(struct1.array, struct2.array)
		assert not np.shares_memory(struct1.array, struct2.array)

	def test_equality(self):
		struct1 = SimpleStruct()
		struct2 = SimpleStruct(np.arange(10))
		struct3 = struct2.copy()

		assert struct1 == struct1
		assert struct2 == struct2
		assert struct3 == struct3
		assert struct1 != struct2
		assert struct1 != struct3
		assert struct2 == struct3

		# Same array contents, but different class
		class SameSize(StructArray):
			x = arrayfield(SimpleStruct.size)

		struct4 = SameSize(struct1.array.copy())
		assert struct4 != struct1

	def test_convert(self):
		"""Test convert() classmethod wraps arrays and passes through instances."""

		# Wraps a numpy array
		arr = np.zeros(SimpleStruct.size)
		struct = SimpleStruct.convert(arr)
		assert isinstance(struct, SimpleStruct)
		assert struct.array is arr

		# Existing instance returned unchanged
		assert SimpleStruct.convert(struct) is struct

		# Wrong type raises TypeError
		for val in ["not an array", 42, None]:
			with pytest.raises(TypeError):
				SimpleStruct.convert(val)  # type: ignore

		# Instance of unrelated StructArray subclass is not accepted
		nested = NestedStruct()
		with pytest.raises(TypeError):
			SimpleStruct.convert(nested)  # type: ignore

		# Array of wrong shape: convert tries to wrap, which raises in __init__
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct.convert(np.zeros(SimpleStruct.size + 1))

	def test_unwrap_array(self):
		"""Test unwrap_array() classmethod returns array given instance, returns array unchanged."""

		# Instance: returns underlying array
		struct = SimpleStruct()
		assert SimpleStruct.unwrap_array(struct) is struct.array

		# Array with correct shape: returned unchanged
		arr = np.zeros(SimpleStruct.size)
		assert SimpleStruct.unwrap_array(arr) is arr

		# Array with wrong shape on subclass raises
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct.unwrap_array(np.zeros(SimpleStruct.size + 1))
		with pytest.raises(ValueError, match="incorrect shape"):
			SimpleStruct.unwrap_array(np.zeros((SimpleStruct.size, 1)))  # type: ignore

		# Wrong type
		for val in ["not an array", 42, None]:
			with pytest.raises(TypeError):
				SimpleStruct.unwrap_array(val)  # type: ignore

		# Unrelated StructArray subclass: not an instance of cls
		nested = NestedStruct()
		with pytest.raises(TypeError):
			SimpleStruct.unwrap_array(nested)  # type: ignore

		# Called on base StructArray: any 1D array passes through, non-1D raises
		arr1d = np.zeros(7)
		assert StructArray.unwrap_array(arr1d) is arr1d

		# Instance of any subclass works
		assert StructArray.unwrap_array(struct) is struct.array
		assert StructArray.unwrap_array(nested) is nested.array

		# Non-1D array on base class raises
		with pytest.raises(ValueError, match="1D"):
			StructArray.unwrap_array(np.zeros((3, 3)))  # type: ignore
		with pytest.raises(ValueError, match="1D"):
			StructArray.unwrap_array(np.zeros(()))  # type: ignore
