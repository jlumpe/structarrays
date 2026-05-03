"""Test the Field class and its subclasses."""

import numpy as np
import pytest

from structarrays import (
	StructArray, ScalarField, ArrayField, StructField, CustomField, arrayfield,
)
from structarrays.utils import Missing
from .common import (
	SimpleStruct, NestedStruct, NestedInner, disjoint_memory, same_memory,
)


class TestStructArrayFields:
	"""Tests for StructArrayFields."""

	def test_basic(self):
		fields = SimpleStruct.fields

		assert fields.size == sum(field.size for field in fields)
		assert fields.by_name == {field.name: field for field in fields}
		assert fields.names() == [field.name for field in fields]

	def test_sequence(self):
		"""Test sequence protocol."""

		fields = SimpleStruct.fields
		fields_list = list(fields)

		# Length matches iteration
		assert len(fields) == len(fields_list)

		# Item access
		for i, field in enumerate(fields):
			assert fields[i] is field
			# Lookup by name also supported
			assert fields[field.name] is field

		# Slicing
		for slice_ in [slice(2), slice(1, 3)]:
			assert fields[slice_] == tuple(fields_list[slice_])


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

	def test_default_wrong_shape(self):
		"""Constructing an ArrayField with default of incorrect shape raises ValueError."""

		# 1D wrong length
		with pytest.raises(ValueError, match="incorrect shape"):
			ArrayField((3,), default=[1, 2])
		with pytest.raises(ValueError, match="incorrect shape"):
			ArrayField((3,), default=np.zeros(4))

		# Wrong number of dimensions
		with pytest.raises(ValueError, match="incorrect shape"):
			ArrayField((3,), default=np.zeros((3, 1)))
		with pytest.raises(ValueError, match="incorrect shape"):
			ArrayField((2, 3), default=np.zeros(6))

		# 2D wrong shape (right total size)
		with pytest.raises(ValueError, match="incorrect shape"):
			ArrayField((2, 3), default=np.zeros((3, 2)))


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
		arr = np.full(NestedStruct.size, 42)
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
		arr = np.zeros(NestedStruct.size)
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

		struct = SimpleStruct(np.full(SimpleStruct.size, -1))
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
		assert f.size == NestedInner.size

	def test_field_passes_kwargs(self):
		f = arrayfield(None, default=99.0)
		assert f.default == 99.0
		f2 = arrayfield(3, default_factory=lambda: np.ones(3))
		assert f2.default_factory is not None
		arr = f2.get_default()
		assert arr is not None
		assert np.array_equal(arr, [1, 1, 1])


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
