"""iPQB boot camp homework assignment.

Author: (Michael) Jared Lumpe


PDB files used as input are assumed to be version 3.3, see
http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
for documentation on the format.
"""

from collections import namedtuple

import math


# Fields of an "ATOM" line in a PDB file
# http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
PDB_ATOM_LINE_FIELDS = [
	(6, 11, int, 'serial'),  # Atom  serial number.
	(12, 16, str, 'name'),  # Atom name.
	(16, 17, str, 'altLoc'),  # Alternate location indicator.
	(17, 20, str, 'resName'),  # Residue name.
	(21, 22, str, 'chainID'),  # Chain identifier.
	(22, 26, int, 'resSeq'),  # Residue sequence number.
	(26, 27, str, 'iCode'),  # Code for insertion of residues.
	(30, 38, float, 'x'),  # Orthogonal coordinates for X in Angstroms.
	(38, 46, float, 'y'),  # Orthogonal coordinates for Y in Angstroms.
	(46, 54, float, 'z'),  # Orthogonal coordinates for Z in Angstroms.
	(54, 60, float, 'occupancy'),  # Occupancy.
	(60, 66, float, 'tempFactor'),  # Temperature factor.
	(76, 78, str, 'element'),  # Element symbol, right-justified.
	(78, 80, str, 'charge'),  # Charge  on the atom.
]


# Named tuple containing parsed data from an ATOM line in a PDB file
PDBAtom = namedtuple('PDBAtom', [
	'serial',
	'name',
	'altLoc',
	'resName',
	'chainID',
	'resSeq',
	'iCode',
	'coord',
	'occupancy',
	'tempFactor',
	'element',
	'charge',
])


# Namde tuple containing parsed data on a PDB residue and its atoms
PDBResidue = namedtuple('PDBResidue', 'name, chainID, seq, atoms')


class Vector3(tuple):
	"""A three dimensional real-valued vector.

	Would be better to use Numpy for these but this seems to be part of the
	assignment. Implements all numeric magic methods that make sense for a
	vector type.

	Dot and cross products can be calculated through the use of the * and @
	(matrix multiplication) operators, respectively:

	>>> Vector3(1, 2, 3) * Vector3(4, 5, 6)
	32.0

	>>> Vector3(1, 2, 3) @ Vector3(4, 5, 6)
	Vector3(1, 2, 3) * Vector3(4, 5, 6)
	"""

	def __new__(cls, *args):

		if len(args) == 0:
			coords = (0, 0, 0)

		elif len(args) == 1:
			if isinstance(args[0], Vector3):
				return args[0]

			coords = list(args[0])
			if len(coords) != 3:
				raise ValueError('')

		elif len(args) == 3:
			coords = args

		else:
			raise TypeError('Constructor takes 0, 1 or 3 positional arguments')

		# Convert to floats
		return super().__new__(cls, map(float, coords))

	@property
	def x(self):
		return self[0]

	@property
	def y(self):
		return self[1]

	@property
	def z(self):
		return self[2]

	def __bool__(self):
		"""Truthy if not the zero vector."""
		return any(c != 0 for c in self)

	def __add__(self, other):
		if isinstance(other, Vector3):
			return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
		else:
			return NotImplemented

	def __sub__(self, other):
		if isinstance(other, Vector3):
			return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
		else:
			return NotImplemented

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			# Multiplication by scalar
			return Vector3(self.x * other, self.y * other, self.z * other)
		elif isinstance(other, Vector3):
			# Dot product with another vector
			return self.dot(other)
		else:
			return NotImplemented

	def __rmul__(self, other):
		if isinstance(other, (int, float)):
			return self * other
		else:
			return NotImplemented

	def __truediv__(self, scalar):
		if isinstance(scalar, (int, float)):
			return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
		else:
			return NotImplemented

	def norm2(self):
		"""Get the vector's squared norm.

		:rtype: float
		"""
		return self.x ** 2 + self.y ** 2 + self.z ** 2

	def norm(self):
		"""Get the vector's magnitude.

		:rtype: float
		"""
		return math.sqrt(self.norm2())

	def __abs__(self):
		return self.norm()

	def __neg__(self):
		return Vector3(-self.x, -self.y, -self.z)

	def __pos__(self):
		return self

	def normalize(self):
		"""Get a normalized version of the vector with unit magnitude.

		:rtype: .Vector3d
		"""
		norm = self.norm()
		if norm == 0:
			raise ValueError("Can't normalize zero vector")
		return self / norm

	def dot(self, other):
		"""Calculate the dot product of this vector with another.

		:type other: .Vector3
		:rtype: float
		"""
		return self.x * other.x + self.y * other.y + self.z * other.z

	def __matmul__(self, other):
		if isinstance(other, Vector3):
			return self.cross(other)
		else:
			return NotImplemented

	def cross(self, other):
		"""Calculate the cross product of this vector with another.

		:type other: .Vector3
		:rtype: float
		"""
		return Vector3(
			self.y * other.z - self.z * other.y,
			self.z * other.x - self.x * other.z,
			self.x * other.y - self.y * other.x,
		)

	def __repr__(self):
		return '{}({}, {}, {})'.format(type(self).__name__, *self)


def parse_pdb_atom(line):
	"""Parse an ATOM line from a PDB file.

	:type line: str
	:rtype: .PDBAtom
	"""
	attrs = dict()

	for begin, end, type_, fieldname in PDB_ATOM_LINE_FIELDS:
		strval = line[begin:end].strip()

		if strval:
			val = type_(strval)
		else:
			val = None

		attrs[fieldname] = val

	# Combine x, y, z, field values into a single attribute
	attrs['coord'] = Vector3(attrs['x'], attrs['y'], attrs['z'])
	del attrs['x'], attrs['y'], attrs['z']

	return PDBAtom(**attrs)


def parse_pdb_chain(fobj):
	"""Lazily parse a chain of residues from a PDB file.

	File must contain only a single chain.

	This is able to parse all ~12k PDB files in the link provided in the
	bootcamp website (no errors, but didn't actually validate output).

	:param fobj: Open file object in text mode. Alternatively, may be any
		iterable of PDB file lines.
	:returns: Generator yielding :class:`.PDBResidue`.
	"""

	currentres = None

	for line in fobj:
		# Ignore non-ATOM lines
		if not line.startswith('ATOM'):
			continue

		# Parse line
		atom = parse_pdb_atom(line.strip())

		# Create a new residue if needed
		if currentres is None or atom.resSeq != currentres.seq:

			if currentres is not None:
				# Check same chain
				if atom.chainID != currentres.chainID:
					raise ValueError('PDB files with multiple chains not supported.')

				yield currentres

			currentres = PDBResidue(atom.resName, atom.chainID, atom.resSeq, [])

		# Add atom to current residue
		currentres.atoms.append(atom)

	if currentres is not None:
		yield currentres
