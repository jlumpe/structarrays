"""Field types."""

from typing import Any, Self, SupportsIndex, overload
from collections.abc import Sequence, Callable
import math
import operator

import numpy as np
from numpy.typing import ArrayLike

from .base import StructArray, Field, Vector
from .utils import Missing, check_shape, format_shape, shape_matches


type _FromArrayFunc[T] = Callable[[Vector[Any]], T]
type _ToArrayFunc[T] = Callable[[T], ArrayLike]


class ScalarField[T](Field[T]):
	"""A single-element field (e.g. a float)."""

	def __init__(self, **kw):
		super().__init__(1, **kw)

	def update(self, **kw) -> Self:
		kw.setdefault('default', self.default)
		kw.setdefault('default_factory', self.default_factory)
		return type(self)(**kw)

	def _get(self, array: Vector[Any]) -> Any:
		return array[self.offset]

	def _set(self, array: Vector[Any], value: T) -> None:
		array[self.offset] = value


class ArrayField(Field[np.ndarray]):
	"""A fixed-shape array field (e.g. a 3-vector or 3x3 matrix)."""

	shape: tuple[int, ...]

	def __init__(self, shape: Sequence[SupportsIndex | None], default: ArrayLike | Missing = Missing(), **kw):

		self.shape = tuple(  # pyright: ignore[reportAttributeAccessIssue]
			-1 if s is None or s < 0 else operator.index(s)
			for s in shape
		)
		size = -1 if any(s < 0 for s in self.shape) else math.prod(self.shape)

		if not isinstance(default, Missing):
			default = np.asarray(default)
			check_shape(default, self.shape, 'default')

		super().__init__(size, default=default, **kw)

	def ndim(self) -> int:
		return len(self.shape)

	def update(self, **kw) -> Self:
		kw.setdefault('shape', self.shape)
		kw.setdefault('default', self.default)
		kw.setdefault('default_factory', self.default_factory)
		return type(self)(**kw)

	def check_override(self, field: Field) -> None:
		field = self._check_override_subclass(field, ArrayField)
		if not shape_matches(field.shape, self.shape):
			clsname = type(self).__name__
			own_shape = format_shape(self.shape)
			field_shape = format_shape(field.shape)
			raise ValueError(
				f'Cannot override {clsname} {self.name!r} of shape {own_shape} with field of shape '
				f'{field_shape}'
			)

	def _from_raw(self, array: Vector[Any]) -> np.ndarray:
		return array.reshape(self.shape)

	def _to_raw(self, value: np.ndarray) -> Any:
		value = np.asarray(value)  # type: ignore
		if value.shape != self.shape:
			raise ValueError(f'Incorrect shape (expected {self.shape}, got {value.shape})')
		return value.flat


class StructField[T: StructArray](Field[T]):
	"""A nested StructArray field."""

	cls: type[T]

	def __init__(self, cls: type[T], **kw):
		if not issubclass(cls, StructArray):
			raise TypeError(f'Expected subclass of StructArray, got {cls}')
		self.cls = cls
		super().__init__(cls.size, **kw)

	def update(self, **kw) -> Self:
		kw.setdefault('cls', self.cls)
		kw.setdefault('default', self.default)
		kw.setdefault('default_factory', self.default_factory)
		return type(self)(**kw)

	def check_override(self, field: Field) -> None:
		field = self._check_override_subclass(field, StructField)
		if not issubclass(field.cls, self.cls):
			msg = 'not a subclass'
		elif not self.cls.is_abstract() and field.cls.size != self.cls.size:
			msg = 'incompatible sizes'
		else:
			msg = None
		if msg is not None:
			raise ValueError(
				f'Cannot override StructField[{self.cls.__name__}] {self.name!r} with '
				f'StructField[{field.cls.__name__}] ({msg})'
			)

	def set_default(self, array: StructArray | Vector[Any]) -> None:
		if self.has_default():
			return super().set_default(array)
		# If no explicit default set on field, use nested StructArray's default values
		if isinstance(array, StructArray):
			array = array.array
		self.get(array).set_defaults()

	def _from_raw(self, array: Vector[Any]) -> T:
		return self.cls(array)

	def _to_raw(self, value: T) -> Any:
		if not isinstance(value, self.cls):
			raise TypeError(f'Expected instance of {self.cls.__name__}, got {type(value)}')
		return value.array


class CustomField[T](Field[T]):
	"""A field with custom getter/setter functions.

	Attributes
	----------
	from_array
		Converter from raw array data to property type.
	to_array
		(Optional) converter from property type to raw array data.
	"""

	from_array: _FromArrayFunc[T]
	to_array: _ToArrayFunc[T] | None

	def __init__(
		self,
		size: int,
		from_array: _FromArrayFunc[T],
		to_array: _ToArrayFunc[T] | None = None,
		**kw,
	):
		if size < 0:
			raise ValueError('Size must be non-negative')
		super().__init__(size, **kw)
		self.from_array = from_array
		self.to_array = to_array

	def update(self, **kw) -> Self:
		kw.setdefault('size', self.size)
		kw.setdefault('from_array', self.from_array)
		kw.setdefault('to_array', self.to_array)
		kw.setdefault('default', self.default)
		kw.setdefault('default_factory', self.default_factory)
		return type(self)(**kw)

	@classmethod
	def wrap(cls, size: int) -> Callable[[_FromArrayFunc[T]], Self]:
		"""
		Return a decorator that takes a function and returns a CustomField instance using the
		function as ``from_array``.
		"""
		def decorator(from_array: _FromArrayFunc[T]) -> Self:
			return cls(size, from_array)
		return decorator

	def _from_raw(self, array: Vector[Any]) -> T:
		return self.from_array(array)

	def _to_raw(self, value: T) -> Vector[Any]:
		if self.to_array is None:
			raise AttributeError(f'Cannot set {type(self).__name__} {self.name!r} with to_array=None')
		array = self.to_array(value)
		return np.asarray(array)


class VoidField(Field[None]):
	"""Field which does not support data access.

	This can be used to create a placeholder field that can be overridden in a subclass.
	"""

	def __init__(self, size: int = -1, **kw):
		super().__init__(-1 if size < 0 else size, **kw)
		if self.has_default():
			raise ValueError('VoidField does not support default values')

	def update(self, **kw) -> Self:
		kw.setdefault('size', self.size)
		return type(self)(**kw)

	def check_override(self, field: Field) -> None:
		if not self.is_abstract() and field.size != self.size:
			if field.is_abstract():
				desc = f'abstract {type(field).__name__}'
			else:
				desc = f'{type(field).__name__} of size {field.size}'
			raise ValueError(f'Cannot override VoidField of size {self.size} with {desc}')

	def _get(self, array: Vector[Any]) -> None:
		return None


@overload
def field(arg: None, **kw) -> ScalarField: ...

@overload
def field(arg: SupportsIndex, **kw) -> ArrayField: ...

@overload
def field(arg: tuple[SupportsIndex, ...], **kw) -> ArrayField: ...

@overload
def field(arg: type[StructArray], **kw) -> StructField: ...

def field(
	arg: None | SupportsIndex | tuple[SupportsIndex, ...] | type[StructArray] = None,
	**kw,
) -> ScalarField | ArrayField | StructField:
	"""Convenience function to create a field descriptor for a StructArray subclass.

	``arg`` can be any of the following:

	* ``None`` (default): creates a :class:`.ScalarField`.
	* ``int``: creates a 1-dimensional :class:`.ArrayField` with the given length.
	* ``tuple`` or sequence of ``int`` or ``None``: creates an N-dimensional :class:`.ArrayField`
	  with the given shape. Dimensions that are ``None`` or negative are considered unspecified, and
	  the field will be considered abstract.
	* ``StructArray`` subclass: creates a :class:`.StructField` for the given subclass.

	Parameters
	----------
	arg
	kw
		Passed to the :class:`.Field` subclass constructor.
	"""
	if arg is None:
		return ScalarField(**kw)

	if isinstance(arg, type) and issubclass(arg, StructArray):
		return StructField(arg, **kw)

	if isinstance(arg, SupportsIndex):
		shape = (arg,)
	else:
		shape = arg
	return ArrayField(shape, **kw)
