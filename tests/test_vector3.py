"""Test the Vector3 class."""

import operator as ops

import pytest
import numpy as np

from ramachandran import Vector3


def check_operator_result(op, *args):
	"""
	Apply an operator to some vectors as Vector3 instances or numpy.ndarrays
	and check that both classes yield identical results.
	"""

	def _convert(obj, converter):
		if np.isscalar(obj):
			return obj
		else:
			return converter(obj)

	vec3_args = [_convert(a, Vector3) for a in args]
	vec3_result = op(*vec3_args)

	numpy_args = [_convert(a, np.asarray) for a in args]
	numpy_result = op(*numpy_args)

	assert np.allclose(vec3_result, numpy_result)


@pytest.fixture
def vecs():
	"""Stack of 10 random 3d vectors."""
	random = np.random.RandomState(0)
	return list(map(Vector3, random.normal(size=(10, 3), scale=10)))


def test_constructor():
	"""Test creating new Vector3 objects."""

	v = Vector3(1, 2, 3)

	# No args creates zero vector
	assert Vector3() == Vector3(0, 0, 0)

	# From x, y, z coordinates
	assert Vector3(1, 2, 3) == v

	# From a sequence
	assert Vector3([1, 2, 3]) == v
	assert Vector3(Vector3(1, 2, 3)) == v
	assert Vector3(range(1, 4)) == v

	# Invalid number of arguments
	with pytest.raises(TypeError):
		Vector3(1, 2)

	with pytest.raises(TypeError):
		Vector3(1, 2, 3, 4)

	# From sequence with improper length
	with pytest.raises(ValueError):
		Vector3([])

	with pytest.raises(ValueError):
		Vector3([1, 2, 3, 4])


def test_bool(vecs):
	"""Test boolean value of vectors."""
	assert not Vector3()

	for v in vecs:
		assert v or all(c == 0 for c in v)


def test_binary_vector_ops(vecs):
	"""Test binary operators on two vectors."""
	for op in [ops.add, ops.sub]:
		for v in vecs:
			for u in vecs:
				check_operator_result(op, u, v)


def test_unary_ops(vecs):
	"""Test unary operators on vectors."""
	for op in [ops.pos, ops.neg]:
		for v in vecs:
			check_operator_result(op, v)


def test_scalar_ops(vecs):
	"""Test operators taking a vector and a scalar."""

	# Multiplication / division
	for v in vecs:
		for s in [1, 1.0, -.353, 15.44]:
			check_operator_result(ops.mul, v, s)
			check_operator_result(ops.mul, s, v)

			check_operator_result(ops.truediv, v, s)
			# No division of scalar by vector

		# Multiplication / division by zero
		assert v * 0 == Vector3(0, 0, 0)
		assert 0 * v == Vector3(0, 0, 0)

		with pytest.raises(ZeroDivisionError):
			v / 0


def test_norm(vecs):
	"""Test vector norm related functions."""

	for v in vecs:
		norm = v.norm()

		assert np.isclose(norm, np.linalg.norm(np.asarray(v)))
		assert np.isclose(v.norm2(), norm ** 2)
		assert abs(v) == norm

		normalized = v.normalize()

		assert np.isclose(normalized.norm(), 1)
		assert np.allclose(normalized * v, norm)


def test_dot(vecs):
	"""Test dot product."""

	for v in vecs:
		for u in vecs:
			dotprod = v.dot(u)

			assert u.dot(v) == dotprod
			assert v * u == dotprod
			assert u * v == dotprod

			assert np.asarray(v).dot(np.asarray(u)) == dotprod


def test_cross(vecs):
	"""Test cross product."""

	for v in vecs:
		for u in vecs:
			crossprod = v.cross(u)

			assert u.cross(v) == -crossprod
			assert v @ u == crossprod
			assert u @ v == -crossprod

			assert np.array_equal(
				np.cross(np.asarray(v), np.asarray(u)),
				crossprod
			)
