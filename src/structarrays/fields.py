"""Field types."""

from typing import Any, Callable, Self, SupportsIndex, cast, overload
import math
import operator

import numpy as np
from numpy.typing import ArrayLike

from .base import StructArray, Field, Vector
from .utils import Missing


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


class ArrayField[S: tuple[int, ...]](Field[np.ndarray[S, Any]]):
	"""A contiguous fixed-shape array field (e.g. a 3-vector or 3x3 matrix)."""

	shape: S

	def __init__(self, shape: S, default: ArrayLike | Missing = Missing(), **kw):
		self.shape = shape
		if not isinstance(default, Missing):
			default = cast(np.ndarray[S, Any], np.asarray(default))
			if default.shape != shape:
				raise ValueError(f'Default has incorrect shape (expected {shape}, got {default.shape})')
		super().__init__(math.prod(shape), default=default, **kw)

	def update(self, **kw) -> Self:
		kw.setdefault('shape', self.shape)
		kw.setdefault('default', self.default)
		kw.setdefault('default_factory', self.default_factory)
		return type(self)(**kw)

	def _from_raw(self, array: Vector[Any]) -> np.ndarray[S, Any]:
		return array.reshape(self.shape)

	def _to_raw(self, value: np.ndarray[S, Any]) -> Any:
		value = np.asarray(value)  # type: ignore
		if value.shape != self.shape:
			raise ValueError(f'Incorrect shape (expected {self.shape}, got {value.shape})')
		return value.flat


class StructField[T: StructArray](Field[T]):
	"""A nested StructArray field."""

	cls: type[T]

	def __init__(self, cls: type[T], **kw):
		self.cls = cls
		super().__init__(cls.size, **kw)

	def update(self, **kw) -> Self:
		kw.setdefault('cls', self.cls)
		kw.setdefault('default', self.default)
		kw.setdefault('default_factory', self.default_factory)
		return type(self)(**kw)

	def set_default(self, array: StructArray | Vector[Any]) -> None:
		if self.has_default():
			return super().set_default(array)
		if isinstance(array, StructArray):
			array = array.array
		self.get(array).set_defaults()

	def _from_raw(self, array: Vector[Any]) -> T:
		return self.cls(array)

	def _to_raw(self, value: T) -> Any:
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


@overload
def field(arg: None, **kw) -> ScalarField: ...

@overload
def field(arg: SupportsIndex, **kw) -> ArrayField[tuple[int]]: ...

@overload
def field(arg: tuple[SupportsIndex, ...], **kw) -> ArrayField[tuple[int, ...]]: ...

@overload
def field(arg: type[StructArray], **kw) -> StructField: ...

def field(
	arg: None | SupportsIndex | tuple[SupportsIndex, ...] | type[StructArray] = None,
	**kw,
) -> ScalarField | ArrayField | StructField:
	"""Convenience function to create a field descriptor for a StructArray subclass.

	``arg`` can be any of the following:

	- ``None`` (default): creates a :class:`.ScalarField`.
	- ``int``: creates a 1-dimensional :class:`.ArrayField` with the given length.
	- ``tuple[int, ...]``: creates an N-dimensional :class:`.ArrayField` with the given shape.
	- ``type[StructArray]``: creates a :class:`.StructField` for the given subclass.

	Parameters
	----------
	arg
	**kw
		Passed to the :class:`.Field` subclass constructor.
	"""
	if arg is None:
		return ScalarField(**kw)

	if isinstance(arg, type) and issubclass(arg, StructArray):
		return StructField(arg, **kw)

	if isinstance(arg, SupportsIndex):
		shape = (operator.index(arg),)
	else:
		shape = tuple(map(operator.index, arg))
	return ArrayField(shape, **kw)
