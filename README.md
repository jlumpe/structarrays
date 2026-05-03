# structarrays

[![Build Status](https://github.com/jlumpe/structarrays/actions/workflows/ci.yml/badge.svg)](https://github.com/jlumpe/structarrays/actions/workflows/ci.yml)

Define structured wrappers around flat Numpy arrays that provide access to the underlying data
through named, typed attributes.

Intended to be used with numerical algorithims like optimizers or ODE integrators that operate on
plain vectors.


## Examples

### ODE integration

Solve an initial value problem simulating the motion of a 2D projectile. Scipy's `solve_ivp()` expects a function that takes the current state as a plain 1D array and returns its derivative in the same format.

Define a `StructArray` subclass with `position` and `velocity` fields (both 2D vectors) to represent the state, and use it internally in the derivative function:

```python
import numpy as np
from scipy.integrate import solve_ivp
from structarrays import StructArray, field


class State(StructArray):
	position = field(2)
	velocity = field(2)


def dynamics(t, y):
	# State
	s = State(y)
	# Derivative, initialized with zeroed float array
	ds = State()
	# This sets the first two elements of ds.array
	ds.position = s.velocity
	# Gravity
	# ds.velocity is a view of ds.array, modifying it modifies the parent
	ds.velocity[1] = -9.81
	# Drag
	ds.velocity -= 0.1 * np.linalg.norm(s.velocity) * s.velocity
	# Return flat Numpy array
	return ds.array


initial = State(position=[0, 0], velocity=[10, 20])
sol = solve_ivp(dynamics, (0, 5), initial.array)

final = State(sol.y[:, -1])
print('final position:', final.position)
print('final velocity:', final.velocity)
```


### Field types

Demonstration of different field types:

```python
import numpy as np
from structarrays import StructArray, field


class Point(StructArray):
	x = field()
	y = field()


class MyStruct(StructArray):
	# Scalar
	scalar = field()
	# 3-element vector
	vector = field(3)
	# 3x3 matrix
	matrix = field((3, 3))
	# Nested StructArray instance
	point = field(Point)
```


```pycon
>>> struct = MyStruct(np.arange(MyStruct.size))
>>> struct.array
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
>>> struct.scalar
np.int64(0)
>>> struct.vector
array([1, 2, 3])
>>> struct.matrix
array([[ 4,  5,  6],
	   [ 7,  8,  9],
	   [10, 11, 12]])
>>> struct.point
<__main__.Point object at 0x7f6427ee82f0>
>>> struct.point.x, struct.point.y
(np.int64(13), np.int64(14))
```


## Field types

### Scalar

Provides access to a single array element. Create with `field()` or `field(None)`.


### Subarray

A writeable view into the the struct's underlying array, optionally reshaped to the given size:

- `field(3)`: 1D array of length 3
- `field((2, 3, 4))`: 3D array with shape `(2, 3, 4)`.


### Nested structure

An instance of another `StructArray` subclass. The array it wraps is a view of the parent's data,
writes to one will affect the other. Create by passing the nested subclass to `field()`.


### Custom field

`CustomField` works similar to `property`, it wraps a getter function `from_array` which converts
from the raw 1D array data to an arbitrary value. Optionally, it can support assignment by providing
a `to_array` function which converts the value to data which can be assigned to the field's array
slice.

```python
from dataclasses import dataclass
from structarrays import StructArray, CustomField


@dataclass
class Point:
	x: float
	y: float

	def to_array(self):
		# Can be anything convertable to a Numpy array
		return [self.x, self.y]

	@classmethod
	def from_array(cls, arr):
		x, y = arr
		return Point(float(x), float(y))


class MyStruct(StructArray):
	point = CustomField(2, Point.from_array, Point.to_array)
```

```pycon
>>> struct = MyStruct([1., 2.])
>>> struct.point
Point(x=1.0, y=2.0)
>>> struct.point = Point(3., 4.)
>>> struct.array
array([3., 4.])
```
