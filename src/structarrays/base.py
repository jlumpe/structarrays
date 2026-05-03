"""
Base classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Self, ClassVar, SupportsIndex, overload
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from .utils import Missing, check_shape


type Vector[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]


class Field[T](ABC):
	"""Descriptor for a field within a StructArray.

	Defines layout (offset, size) and get/set behavior.

	Subclasses must implement :meth:`get` or :meth:`_from_raw`, and can optionally implement
	:meth:`set` or :meth:`_to_raw` to support assignment.

	If neither ``default`` nor ``default_factory`` are provided, the default corresponds to the
	array slice containing all zeros.

	Attributes
	----------
	size
		Number of array elements containing the field's data. Negative means the size is
		unspecified, a class with the field will be considered abstract.
	offset
		Index of the first array element for the field. Negative if the field or ``StructArray``
		instance that contains it is abstract.
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
	_initialized: bool

	def __init__(
		self,
		size: int,
		default: T | Missing = Missing(),
		default_factory: Callable[[], T] | None = None,
	):
		if not isinstance(default, Missing) and default_factory is not None:
			raise ValueError('Cannot specify both default and default_factory')
		self._initialized = False
		self.size = size
		self.offset = -1
		self.name = ''
		self.default = default
		self.default_factory = default_factory

	def _init(self, name: str, offset: int = -1) -> None:
		if self._initialized:
			raise ValueError(f'Field {self.name!r} is already initialized')
		self.name = name
		self.offset = offset
		self._initialized = True

	def _check(self) -> None:
		if not self._initialized:
			raise ValueError('Field is not initialized')
		if self.offset < 0:
			raise ValueError('Field offset not defined (field or field set is abstract)')

	def is_abstract(self) -> bool:
		return self.size < 0

	def check_override(self, field: 'Field') -> None:
		"""Validate field overriding this one in a subclass.

		Raises
		------
		ValueError
			If the field overrides this one in an incompatible way.
		"""
		self._check_override_subclass(field)

	@overload
	def _check_override_subclass(self, field: Field, cls: None = None) -> Self: ...

	@overload
	def _check_override_subclass[F: Field](self, field: Field, cls: type[F]) -> F: ...

	def _check_override_subclass(self, field: Field, cls: type[Field] | None = None) -> Field:
		"""Check overriding field is an instance of the same class (or the given class).

		Return argument as the given type for the benefit of the type checker.
		"""
		if cls is None:
			cls = type(self)
		if not isinstance(field, cls):
			raise ValueError(
				f'Cannot override field {self.name!r} of type {cls.__name__} '
				f'with a field of type {type(field).__name__}'
			)
		return field

	def copy(self) -> Self:
		"""Create a copy of the field."""
		return self.update()

	@abstractmethod
	def update(self) -> Self:
		"""Create a copy of the field with different attributes.

		Keyword arguments should match constructor.
		"""

	# ----------------------------------------- Defaults ----------------------------------------- #

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

	# ---------------------------------------- Data access --------------------------------------- #

	def get_raw[E: np.generic](self, array: StructArray[E] | Vector[E]) -> Vector[E]:
		"""Get the raw array slice for this field from the parent array."""
		self._check()
		if isinstance(array, StructArray):
			array = array.array
		return array[self.offset:self.offset + self.size]

	def set_raw(self, array: StructArray | Vector[Any], value: Any) -> None:
		"""Write the raw array slice for this field in the parent array."""
		self._check()
		if isinstance(array, StructArray):
			array = array.array
		array[self.offset:self.offset + self.size] = value

	def set_default(self, array: StructArray | Vector[Any]) -> None:
		"""Apply this field's default to the given struct instance or array.

		If no explicit default is set, assigns zeros.
		"""
		if isinstance(array, StructArray):
			array = array.array
		default = self.get_default()
		if not isinstance(default, Missing):
			self.set(array, default)
		else:
			self._zero(array)

	def _zero(self, array: Vector[Any]) -> None:
		"""Set the field's slice to zero."""
		self._check()
		array[self.offset:self.offset + self.size] = 0

	# ---------------------------------------- Data access --------------------------------------- #

	def get(self, array: Vector[Any]) -> T:
		"""Get the value from the parent array."""
		field_arr = self.get_raw(array)
		try:
			value = self._from_raw(field_arr)
		except NotImplementedError:
			raise TypeError(f'Field subclass {self.__class__.__name__} must implement get() or _from_raw()') from None
		return value

	def set(self, array: Vector[Any], value: T) -> None:
		"""Set the value in the parent array."""
		try:
			slice_value = self._to_raw(value)
		except NotImplementedError:
			raise self._assignment_unsupported() from None

		self.set_raw(array, slice_value)

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

	@overload
	def __get__(self, instance: StructArray, owner=None) -> T: ...

	@overload
	def __get__(self, instance: None, owner=None) -> Self: ...

	def __get__(self, instance: StructArray | None, owner=None) -> T | Self:
		self._check()
		if instance is None:
			return self
		return self.get(instance.array)

	def __set__(self, instance: StructArray | None, value: T) -> None:
		self._check()
		if instance is None:
			raise AttributeError('Cannot set Field attribute on class')
		self.set(instance.array, value)

	def _from_raw(self, array: Vector[Any]) -> T:
		"""Get the value from the field's slice of the parent array.

		Not required, but used in default implementation of ``get()``.
		"""
		raise NotImplementedError()

	def _to_raw(self, value: T) -> Any:
		"""Convert the value to something suitable to assign to the field's slice.

		Not required, but used in default implementation of ``set()``.
		"""
		raise NotImplementedError()

	@classmethod
	def _assignment_unsupported(cls) -> AttributeError:
		return AttributeError(f'Field of type {cls.__name__} does not support assignment')


@dataclass(frozen=True, repr=False)
class StructArrayFields(Sequence[Field]):
	"""Sequence of fields for a StructArray.

	Acts as an immutable sequence of field instances, but also supports looking up fields by name.
	"""

	size: int
	by_name: Mapping[str, Field]
	_fields: tuple[Field, ...]

	def __init__(self, fields: Mapping[str, Field]):
		"""Create from a dictionary of uninitialized fields. The fields will be initialized with
		names and offsets (the latter only if all are non-abstract).
		"""
		fields = dict(fields)

		# Empty -> abstract
		is_abstract = not fields or any(field.is_abstract() for field in fields.values())

		object.__setattr__(self, '_fields', tuple(fields.values()))
		object.__setattr__(self, 'by_name', MappingProxyType(fields))

		if is_abstract:
			object.__setattr__(self, 'size', -1)

			for name, field in fields.items():
				field._init(name)

		else:
			offset = 0
			for name, field in fields.items():
				field._init(name, offset)
				offset += field.size

			object.__setattr__(self, 'size', offset)

	def is_abstract(self) -> bool:
		return self.size < 0

	def names(self) -> list[str]:
		return [field.name for field in self._fields]

	def abstract_fields(self) -> list[Field]:
		return [field for field in self._fields if field.is_abstract()]

	def update(self, fields: Mapping[str, Field]) -> StructArrayFields:
		"""Update with new or overriding field definitions, as in class inheritance.

		New fields will be appended to the end. Overrides to existing fields will be checked for
		compatibility.
		"""

		merged = dict()

		# Existing fields - check for overrides, otherwise copy from self
		for field in self:
			if field.name in fields:
				field.check_override(fields[field.name])
				merged[field.name] = fields[field.name]
			else:
				merged[field.name] = field.copy()

		# Append new fields
		for name, field in fields.items():
			if name not in self.by_name:
				merged[name] = field

		return StructArrayFields(merged)

	# ------------------------------------ Sequence interface ------------------------------------ #

	def __len__(self) -> int:
		return len(self._fields)

	@overload
	def __getitem__(self, index: SupportsIndex | str) -> Field: ...

	@overload
	def __getitem__(self, index: slice) -> tuple[Field, ...]: ...

	def __getitem__(self, index: SupportsIndex | str | slice) -> Field | tuple[Field, ...]:
		if isinstance(index, str):
			return self.by_name[index]
		return self._fields[index]

	def __iter__(self) -> Iterator[Field]:
		return iter(self._fields)


class StructArray[T: np.generic]:
	"""Structured data backed by a contiguous 1D numpy array.

	Subclasses contain one or more fields (as :class:`Field` instances) which each correspond to a
	contiguous block of elements in the backing array. Attribute access through the fields
	transforms the raw array data into the appropriate type.

	Attributes
	----------
	fields
		Field descriptors (class attribute).
	size
		Total size of the array (class attribute).
	array
		The flat 1D array storing all field data.
	"""

	fields: ClassVar[StructArrayFields] = StructArrayFields({})
	size: ClassVar[int] = -1

	array: Vector[T]

	def __init_subclass__(cls):

		# Determine parent StructArray class
		parent = None

		for base in cls.__bases__:
			if issubclass(base, StructArray):
				if parent is not None:
					raise TypeError('Multiple inheritance of StructArray classes is not supported')
				parent = base

		# Shouldn't be possible?
		assert parent is not None

		# Collect fields
		fields = dict()

		for name, val in cls.__dict__.items():
			if isinstance(val, Field):
				# Clashes with method or base attribute
				if hasattr(StructArray, name) or name in StructArray.__annotations__:
					raise ValueError(f'Invalid field name {name!r}')
				fields[name] = val

		cls.fields = parent.fields.update(fields)
		cls.size = cls.fields.size

	def __init__(self, array: Vector[T] | ArrayLike | None = None, /, **kw):
		"""
		Parameters
		----------
		array
			Array to wrap. If ``None``, allocates a zeroed array and applies field defaults.
		"""
		if self.is_abstract():
			raise TypeError(f'Cannot instantiate abstract StructArray subclass {type(self).__name__}')

		if array is None:
			self.array = np.zeros(self.size)  # type: ignore
			self.set_defaults()

		else:
			array = np.asarray(array)
			check_shape(array, (self.size,))
			self.array = array

		# Set field values
		for name, value in kw.items():
			if name not in self.fields.by_name:
				raise ValueError(f'Invalid field name {name!r}')
			field = self.fields[name]
			field.set(self.array, value)

	@classmethod
	def is_abstract(cls) -> bool:
		return cls.size < 0

	def set_defaults(self) -> None:
		"""Reset all field values to their defaults."""
		for field in self.fields:
			field.set_default(self)

	@classmethod
	def convert[S: StructArray[Any]](cls: type[S], obj: Vector[T] | S) -> S:
		"""Wrap an array in an instance of the class, or return an existing instance unchanged."""
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
			# Abstract: just check 1D
			if cls.is_abstract():
				if obj.ndim != 1:
					raise ValueError('Expected 1D array')
			else:
				check_shape(obj, (cls.size,))
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
			field.name: field.get_raw(self) if raw else field.__get__(self)
			for field in self.fields
		}

	def copy(self) -> Self:
		return type(self)(self.array.copy())

	def __eq__(self, other: Any) -> bool:
		# Check type exactly equal, ok due to inheritance being disallowed
		return type(self) is type(other) and np.array_equal(self.array, other.array)

	# Mutable, explicitly disallow hashing
	__hash__ = None  # pyright: ignore[reportAssignmentType]
