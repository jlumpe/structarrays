"""
Base classes.
"""

from typing import Any, Callable, Self, ClassVar, overload
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from .utils import Missing


type Vector[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]


class StructArray[T: np.generic]:
	"""Structured data backed by a contiguous 1D numpy array.

	Subclasses contain one or more fields (as :class:`Field` instances) which each correspond to a
	contiguous block of elements in the backing array. Attribute access through the fields
	transforms the raw array data into the appropriate type.

	Attributes
	----------
	_fields_
		Tuple of field descriptors.
	_fields_dict_
		Mapping of field names to descriptors.
	_size_
		Total size of the array.
	array
		The flat 1D array storing all field data.
	"""

	_fields_: ClassVar[tuple['Field', ...]] = ()
	_fields_dict_: ClassVar[Mapping[str, 'Field']] = MappingProxyType({})
	_size_: ClassVar[int] = 0

	array: Vector[T]

	def __init_subclass__(cls):

		# Support multiple inheritance of other types, but only direct inheritance from StructArray
		inherit_ok = False
		for base in cls.__bases__:
			# Direct child class
			if base is StructArray:
				inherit_ok = True
			# Also not indirect - could happen with above through diamond inheritance
			elif StructArray in base.__mro__:
				inherit_ok = False
				break
		if not inherit_ok:
			raise TypeError(f'{cls.__name__} must inherit directly from StructArray')

		fields = []
		offset = 0

		for name, val in cls.__dict__.items():
			if isinstance(val, Field):
				# Clashes with method or base attribute
				if hasattr(StructArray, name) or name in StructArray.__annotations__:
					raise ValueError(f'Invalid field name {name!r}')
				if val._initialized():
					raise ValueError(f'Field {name!r} already initialized')
				val.offset = offset  # pyright: ignore[reportAttributeAccessIssue]
				val.name = name  # pyright: ignore[reportAttributeAccessIssue]
				offset += val.size
				fields.append(val)

		cls._fields_ = tuple(fields)
		cls._fields_dict_ = MappingProxyType({field.name: field for field in fields})
		cls._size_ = offset

	def __init__(self, array: Vector[T] | ArrayLike | None = None, /, **kw):
		"""
		Parameters
		----------
		array
			Array to wrap. If ``None``, allocates a zeroed array and applies field defaults.
		"""
		if array is None:
			self.array = np.zeros(self._size_)  # type: ignore
			self.set_defaults()

		else:
			array = np.asarray(array)
			self._check_shape(array)
			self.array = array

		# Set field values
		for name, value in kw.items():
			if name not in self._fields_dict_:
				raise ValueError(f'Invalid field name {name!r}')
			field = self._fields_dict_[name]
			field.__set__(self, value)

	def set_defaults(self) -> None:
		"""Reset all field values to their defaults."""
		for field in self._fields_:
			field.set_default(self)

	@classmethod
	def _check_shape(cls: type[Self], array: Vector[T]) -> None:
		if array.shape != (cls._size_,):
			raise ValueError(f'Array has incorrect shape (expected {cls._size_}, got {array.shape})')

	@classmethod
	def convert[S: StructArray[Any]](cls: type[S], obj: Vector[T] | S) -> S:
		"""Wrap the an array in an instance of the class, or return an existing instance unchanged."""
		if isinstance(obj, np.ndarray):
			return cls(obj)
		if isinstance(obj, cls):
			return obj
		raise TypeError(f'Expected ndarray or {cls.__name__}')

	@classmethod
	def unwrap_array(cls, obj: Vector | Self) -> Vector[Any]:
		"""
		Opposite of ``convert()``: unwrap array given instance, given an array return it unchanged.

		Unless called on base class, checks array size matches the class size.
		"""
		if isinstance(obj, np.ndarray):
			if cls is StructArray:
				if obj.ndim != 1:
					raise ValueError('Expected 1D array')
			else:
				cls._check_shape(obj)
			return obj
		elif isinstance(obj, cls):
			return obj.array
		raise TypeError(f'Expected ndarray or {cls.__name__}')

	def asdict(self, raw: bool = False) -> dict[str, Any]:
		"""Get a mapping from field names to their values.

		Parameters
		----------
		raw
			If ``True``, return the raw array views instead of the converted values.
		"""
		return {
			name: field.get_raw(self) if raw else field.__get__(self)
			for name, field in self._fields_dict_.items()
		}

	def copy(self) -> Self:
		return type(self)(self.array.copy())

	def __eq__(self, other: Any) -> bool:
		# Check type exactly equal, ok due to inheritance being disallowed
		return type(self) is type(other) and np.array_equal(self.array, other.array)

	# Mutable, explicitly disallow hashing
	__hash__ = None  # pyright: ignore[reportAssignmentType]


class Field[T]:
	"""Descriptor for a field within a StructArray.

	Defines layout (offset, size) and get/set behavior.

	Subclasses must implement :meth:`_get` or :meth:`_from_raw`, and can optionally implement
	:meth:`_set` or :meth:`_to_raw` to support assignment.

	If neither ``default`` nor ``default_factory`` are provided, the default corresponds to the
	array slice containing all zeros.

	Attributes
	----------
	size
		Number of array elements containing the field's data.
	offset
		Index of the first array element for the field.
	name
		Name of the field.
	default
		Default value for the field.
	default_factory
		Callable to generate a default value for the field.
	"""

	size: int
	offset: int
	name: str
	default: T | Missing
	default_factory: Callable[[], T] | None

	def __init__(
		self,
		size: int,
		default: T | Missing = Missing(),
		default_factory: Callable[[], T] | None = None,
	):
		if not isinstance(default, Missing) and default_factory is not None:
			raise ValueError('Cannot specify both default and default_factory')
		self.size = size
		self.offset = -1
		self.name = ''
		self.default = default
		self.default_factory = default_factory

	def _initialized(self) -> bool:
		return self.offset >= 0

	def get_raw[E: np.generic](self, array: StructArray[E] | Vector[E]) -> Vector[E]:
		"""Get the raw array slice for this field from the parent array."""
		if isinstance(array, StructArray):
			array = array.array
		return array[self.offset:self.offset + self.size]

	def set_raw(self, array: StructArray | Vector[Any], value: Any) -> None:
		"""Write the raw array slice for this field in the parent array."""
		if isinstance(array, StructArray):
			array = array.array
		array[self.offset:self.offset + self.size] = value

	def has_default(self) -> bool:
		"""Whether the field has an explicit default set."""
		return not (isinstance(self.default, Missing) and self.default_factory is None)

	def get_default(self) -> T | Missing:
		"""Get the default value (from ``default`` or ``default_factory``)."""
		if not isinstance(self.default, Missing):
			return self.default
		if self.default_factory is not None:
			return self.default_factory()
		return Missing()

	def set_default(self, array: StructArray | Vector[Any]) -> None:
		"""Apply this field's default to the given struct instance or array.

		If no explicit default is set, assigns zeros.
		"""
		if isinstance(array, StructArray):
			array = array.array
		default = self.get_default()
		if not isinstance(default, Missing):
			self._set(array, default)
		else:
			self._zero(array)

	def _zero(self, array: Vector[Any]) -> None:
		"""Set the field's slice to zero."""
		array[self.offset:self.offset + self.size] = 0

	@overload
	def __get__(self, instance: StructArray, owner=None) -> T: ...

	@overload
	def __get__(self, instance: None, owner=None) -> Self: ...

	def __get__(self, instance: StructArray | None, owner=None) -> T | Self:
		if instance is None:
			return self
		return self._get(instance.array)

	def __set__(self, instance: StructArray | None, value: T) -> None:
		if instance is None:
			raise AttributeError('Cannot set Field attribute on class')
		self._set(instance.array, value)

	def _get(self, array: Vector[Any]) -> T:
		"""Get the value from the parent array."""
		field_arr = self.get_raw(array)
		try:
			value = self._from_raw(field_arr)
		except NotImplementedError:
			raise TypeError(f'Field subclass {self.__class__.__name__} must implement _get() or _from_raw()') from None
		return value

	def _from_raw(self, array: Vector[Any]) -> T:
		"""Get the value from the field's slice of the parent array.

		Not required, but used in default implementation of ``_get()``.
		"""
		raise NotImplementedError()

	def _set(self, array: Vector[Any], value: T) -> None:
		"""Set the value in the parent array."""
		try:
			slice_value = self._to_raw(value)
		except NotImplementedError:
			raise self._assignment_unsupported() from None

		self.set_raw(array, slice_value)

	def _to_raw(self, value: T) -> Any:
		"""Convert the value to something suitable to assign to the field's slice.

		Not required, but used in default implementation of ``_set()``.
		"""
		raise NotImplementedError()

	@classmethod
	def _assignment_unsupported(cls) -> AttributeError:
		return AttributeError(f'Field of type {cls.__name__} does not support assignment')
