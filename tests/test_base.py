"""Test core functionality."""

import numpy as np
import pytest

from structarrays import (
	StructArray, Field, ScalarField, ArrayField, StructField, CustomField, arrayfield,
)
from structarrays.utils import Missing


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
#                                              Classes                                             #
# ------------------------------------------------------------------------------------------------ #

class SimpleStruct(StructArray):
	"""Struct with scalar, vector, and matrix fields."""

	x = arrayfield(None, default=1.5)
	y = arrayfield(3)
	z = arrayfield((2, 3), default=[[11, 12, 13], [21, 22, 23]])


class NestedInner(StructArray):
	"""Inner struct for nesting tests."""

	a = arrayfield(None, default=10.0)
	b = arrayfield(2)


class NestedStruct(StructArray):
	"""Struct with nested StructArray field."""

	inner = arrayfield(NestedInner)
	z = arrayfield(None, default=3.0)


class StructWithDefaults(StructArray):
	"""Struct with default_factory."""

	vec = arrayfield(3, default_factory=lambda: np.array([1, 2, 3]))


# ------------------------------------------------------------------------------------------------ #
#                                               Tests                                              #
# ------------------------------------------------------------------------------------------------ #

class TestStructArray:
	"""Tests for StructArray base class."""

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

	def test_asdict(self):
		"""Test asdict() method with raw=False."""
		struct = SimpleStruct(np.arange(10, 20))
		d = struct.asdict()
		assert d.keys() == SimpleStruct._fields_dict_.keys()
		for k, v in d.items():
			assert array_scalar_equal(v, getattr(struct, k))

	def test_asdict_raw(self):
		"""Test asdict() method with raw=True."""
		struct = SimpleStruct(np.arange(10, 20))
		arrs = struct.asdict(raw=True)
		assert arrs.keys() == SimpleStruct._fields_dict_.keys()
		for k, v in arrs.items():
			field = getattr(SimpleStruct, k)
			assert same_memory(v, field.get_raw(struct))

	def test_class_metadata(self):
		"""Test _fields_, _fields_dict_, _size_ are set correctly."""
		# _size_
		assert SimpleStruct._size_ == 10
		# _fields_
		assert isinstance(SimpleStruct._fields_, tuple)
		assert len(SimpleStruct._fields_) == 3
		for i, name in enumerate(['x', 'y', 'z']):
			assert SimpleStruct._fields_[i] is getattr(SimpleStruct, name)
		# _fields_dict_
		assert SimpleStruct._fields_dict_.keys() == {'x', 'y', 'z'}
		for k, v in SimpleStruct._fields_dict_.items():
			assert v is getattr(SimpleStruct, k)

	def test_copy(self):
		struct1 = SimpleStruct(np.arange(10))
		struct2 = struct1.copy()
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

		assert struct1 is not None

		# Same array contents, but different class
		class SameSize(StructArray):
			x = arrayfield(SimpleStruct._size_)

		struct4 = SameSize(struct1.array.copy())
		assert struct4 != struct1


class TestScalarField:
	"""Tests for ScalarField."""

	def test_get_set(self):
		struct = SimpleStruct()
		# Get (default)
		assert struct.x == 1.5
		# Reflects array
		struct.array[0] = 3
		assert struct.x == 3
		# Set
		struct.x = 42.0
		assert struct.x == 42.0
		assert struct.array[0] == 42.0


class TestArrayField:
	"""Tests for ArrayField.
	"""

	def test_get_set(self):

		struct = SimpleStruct()

		# Vector field

		# Get default
		assert np.array_equal(struct.y, [0, 0, 0])

		# Reflects array
		yval = [1, 2, 3]
		struct.array[1:4] = yval
		assert np.array_equal(struct.y, yval)

		# Set directly
		yval = [11, 12, 13]
		struct.y = yval  # type: ignore
		assert np.array_equal(struct.y, yval)
		assert np.array_equal(struct.array[1:4], yval)

		# Matrix field

		# Get default
		assert np.array_equal(struct.z, [[11, 12, 13], [21, 22, 23]])

		# Reflects array
		struct.array[4:10] = np.arange(0, 6)
		assert np.array_equal(struct.z, [[0, 1, 2], [3, 4, 5]])

		# Set directly
		zval = [[14, 15, 16], [17, 18, 19]]
		struct.z = zval  # type: ignore
		assert np.array_equal(struct.z, zval)
		assert np.array_equal(struct.array[4:10], np.arange(14, 20))

	def test_wrong_shape(self):
		s = SimpleStruct()
		# 1D, wrong length
		with pytest.raises(ValueError, match="Incorrect shape"):
			s.y = [1, 2]   # type: ignore
		# 2D, wrong shape
		with pytest.raises(ValueError, match="Incorrect shape"):
			s.z = np.zeros((3, 2))  # type: ignore
		# Wrong dims
		with pytest.raises(ValueError, match="Incorrect shape"):
			s.z = s.y  # type: ignore
		with pytest.raises(ValueError, match="Incorrect shape"):
			s.y = s.z  # type: ignore


# -----------------------------------------------------------------------------
# StructField tests
# -----------------------------------------------------------------------------

class TestStructField:
	"""Tests for StructField (nested StructArray)."""

	def test_implicit_default(self):
		"""Test that default values are applied in nested struct's fields."""

		# No arguments - new array created with defaults applied
		struct1 = NestedStruct()
		assert struct1.inner.a == 10.0
		assert np.array_equal(struct1.inner.b, [0, 0])
		assert struct1.z == 3.0

		# With array - not modified
		arr = np.full(NestedStruct._size_, 42)
		struct2 = NestedStruct(arr)
		assert struct2.inner.a == 42
		assert np.array_equal(struct2.inner.b, [42, 42])
		assert struct2.z == 42

		# Apply defaults to existing instance
		struct2.set_defaults()
		assert struct2.inner.a == 10.0
		assert np.array_equal(struct2.inner.b, [0, 0])
		assert struct2.z == 3.0

	def test_explicit_default(self):
		"""Test supplying an explicit default for the field."""

		thedefault = NestedInner(a=1, b=[2, 3])

		class WithDefault(StructArray):
			inner = arrayfield(NestedInner, default=thedefault)

		struct1 = WithDefault()
		assert struct1.inner == thedefault
		# Right array values, but doesn't share memory
		assert disjoint_memory(struct1.inner, thedefault)

		class WithDefaultFactory(StructArray):
			inner = arrayfield(NestedInner, default_factory=thedefault.copy)

		struct2 = WithDefaultFactory()
		assert struct2.inner == thedefault
		assert disjoint_memory(struct2.inner, thedefault)

	def test_nested_modify(self):
		s = NestedStruct()
		s.inner.a = 99.0
		s.inner.b = [1, 2]
		assert s.inner.a == 99.0
		assert np.array_equal(s.inner.b, [1, 2])
		assert s.inner.array.shape == (3,)  # 1 + 2

	def test_nested_is_view(self):
		"""Nested struct shares parent array storage."""
		arr = np.zeros(NestedStruct._size_)
		struct = NestedStruct(arr)

		# Set nested field, read parent array
		struct.inner.a = 42
		assert arr[0] == 42

		# Set parent array, read nested field
		arr[1] = 52
		assert struct.inner.b[0] == 52


# -----------------------------------------------------------------------------
# CustomField tests
# -----------------------------------------------------------------------------


def _tuple_from_array(arr):
	return tuple(arr.tolist())


def _sum_from_array(arr):
	return sum(arr)


def _triple_to_array(val):
	return np.array([val, val, val])


class ReadOnlyStruct(StructArray):
	"""Struct with read-only CustomField (no to_array)."""
	custom_readonly = CustomField(3, from_array=_tuple_from_array, to_array=None)


class RWStruct(StructArray):
	"""Struct with read-write CustomField."""
	custom_rw = CustomField(
		3, from_array=_sum_from_array, to_array=_triple_to_array
	)


class TestCustomField:
	"""Tests for CustomField with custom from_array/to_array."""

	def test_readonly_custom_field(self):
		"""CustomField with to_array=None is read-only."""
		arr = np.array([1.0, 2.0, 3.0])
		s = ReadOnlyStruct(arr)
		assert s.custom_readonly == (1.0, 2.0, 3.0)
		with pytest.raises(AttributeError, match='Cannot set CustomField'):
			s.custom_readonly = (4, 5, 6)

	def test_read_write_custom_field(self):
		"""CustomField with to_array supports both get and set."""
		s = RWStruct()
		s.custom_rw = 5
		assert s.custom_rw == 15  # 5+5+5
		assert np.array_equal(s.array, [5, 5, 5])

	def test_wrap_decorator(self):
		"""CustomField.wrap creates field from decorated function."""
		@CustomField.wrap(2)
		def from_arr(arr):
			return arr[0] + arr[1]

		assert isinstance(from_arr, CustomField)
		assert from_arr.size == 2
		assert from_arr.from_array is not None


# -----------------------------------------------------------------------------
# Field descriptor tests
# -----------------------------------------------------------------------------


class TestFieldDescriptor:
	"""Tests for Field descriptor protocol."""

	def test_get_on_class_returns_descriptor(self):
		"""Accessing field on class returns the descriptor, not value."""
		f = SimpleStruct.y
		assert isinstance(f, ArrayField)

	@pytest.mark.parametrize('pass_array', [True, False])
	def test_get_set_raw(self, pass_array: bool):
		"""Test get_raw() and set_raw() methods."""

		struct = SimpleStruct(np.full(SimpleStruct._size_, -1))
		arg = struct.array if pass_array else struct

		# Set
		SimpleStruct.x.set_raw(arg, 0)
		SimpleStruct.y.set_raw(arg, [1, 2, 3])
		SimpleStruct.z.set_raw(arg, [4, 5, 6, 7, 8, 9])
		assert np.array_equal(struct.array, np.arange(10))

		# Get
		assert same_memory(SimpleStruct.x.get_raw(arg), struct.array[0:1])
		assert same_memory(SimpleStruct.y.get_raw(arg), struct.array[1:4])
		assert same_memory(SimpleStruct.z.get_raw(arg), struct.array[4:10])


# -----------------------------------------------------------------------------
# field() convenience function tests
# -----------------------------------------------------------------------------


class TestFieldFunction:
	"""Tests for the field() convenience function."""

	def test_field_none_returns_scalar(self):
		f = arrayfield(None)
		assert isinstance(f, ScalarField)
		assert f.size == 1

	def test_field_int_returns_1d_array(self):
		f = arrayfield(5)
		assert isinstance(f, ArrayField)
		assert f.shape == (5,)
		assert f.size == 5

	def test_field_tuple_returns_nd_array(self):
		f = arrayfield((2, 3))
		assert isinstance(f, ArrayField)
		assert f.shape == (2, 3)
		assert f.size == 6

	def test_field_struct_type_returns_struct_field(self):
		f = arrayfield(NestedInner)
		assert isinstance(f, StructField)
		assert f.cls is NestedInner
		assert f.size == NestedInner._size_

	def test_field_passes_kwargs(self):
		f = arrayfield(None, default=99.0)
		assert f.default == 99.0
		f2 = arrayfield(3, default_factory=lambda: np.ones(3))
		assert f2.default_factory is not None
		arr = f2.get_default()
		assert arr is not None
		assert np.array_equal(arr, [1, 1, 1])


# -----------------------------------------------------------------------------
# default_factory tests
# -----------------------------------------------------------------------------


class TestDefaultFactory:
	"""Tests for default_factory on fields."""

	def test_default_factory_applied_on_init(self):
		s = StructWithDefaults()
		assert np.array_equal(s.vec, [1, 2, 3])

	def test_default_factory_called_per_instance(self):
		s1 = StructWithDefaults()
		s2 = StructWithDefaults()
		# Each gets its own array from the factory
		assert not np.shares_memory(s1.vec, s2.vec)


# -----------------------------------------------------------------------------
# Invalid field name test
# -----------------------------------------------------------------------------


class TestInvalidFieldName:
	"""Test that reserved/invalid field names are rejected."""

	def test_field_name_collision_with_structarray_attr_raises(self):
		"""Field name that shadows StructArray attribute raises ValueError."""
		with pytest.raises(ValueError, match="Invalid field name"):

			class BadStruct(StructArray):
				_fields_ = arrayfield(1)  # type: ignore


# -----------------------------------------------------------------------------
# Field.get_default / set_default tests
# -----------------------------------------------------------------------------

class TestFieldDefaults:
	"""Tests for Field default handling."""

	def test_get_default_from_value(self):
		f = ScalarField(default=42.0)
		assert f.get_default() == 42.0

	def test_get_default_from_factory(self):
		f = ScalarField(default_factory=lambda: 99)
		assert f.get_default() == 99

	def test_get_default_missing(self):
		f = ScalarField()
		assert f.get_default() is Missing()
