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


def dihedral_calculator():
	"""
	Coroutine/generator which lazily calculates dihedral angles for each set of
	4 contiguous points in a sequence.

	The generator must be sent points one at a time. Points May be
	:class:`.Vector3` or any sequence of x, y, and z coordinates. The yielded
	value will be the dihedral angle for the sent point and the previous three,
	or None if less than four points have been sent so far.

	Angles are in radians in the range (-pi, pi).

	Source for method: https://math.stackexchange.com/a/47084/252073
	"""

	# Prime with first 3 points
	p1 = Vector3((yield None))
	p2 = Vector3((yield None))
	p3 = Vector3((yield None))

	# Set up for first angle
	lastpoint = p3
	lastdisp = p3 - p2
	lastnormal = ((p2 - p1) @ lastdisp).normalize()

	angle = None

	# For each point starting with the 4th, we can compute a new angle
	while True:

		# Yield the last angle (None the first time), get the next point
		nextpoint = Vector3((yield angle))

		# Displacement from previous point to current
		nextdisp = nextpoint - lastpoint

		# Normal vector to plane containing last 3 points
		nextnormal = (lastdisp @ nextdisp).normalize()

		# This one's complicated... see step 3 in source.
		x = lastnormal * nextnormal
		y = (lastnormal @ lastdisp.normalize()) * nextnormal
		angle = -math.atan2(y, x)

		# Current values used as previous in next loop
		lastpoint = nextpoint
		lastdisp = nextdisp
		lastnormal = nextnormal


def calc_dihedrals(points):
	"""Lazily calculate dihedral angles for a set of points.

	Angles are in radians in the range (-pi, pi).

	:param points: Iterable of 3D points, as :class:`.Vector3` or any other
		sequences of x, y, z coordinates.
	:returns: Generator yielding dihedral angles for each contiguous set of 4
		points.
	"""
	piter = iter(points)

	calculator = dihedral_calculator()
	calculator.send(None)

	for i in range(3):
		calculator.send(next(piter))

	for point in piter:
		yield calculator.send(point)


def get_backbone_atoms(residue, strict=False):
	"""Get the N, CA, and C atoms from an amino acid residue.

	:param residue: Amino acid residue to get backbone atoms for.
	:type residue: .PDBResidue
	:param bool strict: Raise an exception if an atom cannot be found.
	:returns: ``(N, CA, C)`` atom tuple.
	:rtype: tuple[.PDBAtom]

	:raises ValueError: If ``strict`` is True and the residue is missing one of
		the atom types or has multiple instances of it.
	"""

	names = ['N', 'CA', 'C']
	atoms = [None] * len(names)

	# Find atoms
	for atom in residue.atoms:
		for i, name in enumerate(names):
			if atom.name == name:
				if strict and atoms[i] is not None:
					raise ValueError('Duplicate {} atom in residue'.format(name))

				atoms[i] = atom
				break

	# Check all found
	if strict:
		for i, name in enumerate(names):
			if atoms[i] is None:
				raise ValueError('No {} atom found in residue'.format(name))

	return tuple(atoms)


def calc_torsion(residues, include_residue=False, include_omega=False):
	"""Lazily calculate phi/psi torsion angles for an amino acid sequence.

	This function attempts to handle invalid or non-contiguous residues
	gracefully, yielding None for any dihedral angles derived from sets of atoms
	that come from these.

	:param residues: Iterable of amino acid residues as :class:`.PDBResidue`.
	:param bool include_residue: Also yield residues along with angles.
	:param bool include_omega: Also yield values for the omega angle.

	:returns: Generator yielding ``(residue, omega, phi, psi)`` tuples, with
		``residue`` and ``omega`` omitted depending on the values of the
		``include_residue`` and ``include_omega`` parameters.
	"""

	last_residue = None
	last_contiguous = True
	last_valid = False

	last_omega = None
	last_phi = None

	def yield_vals(residue, omega, phi, psi):
		angles = (omega, phi, psi) if include_omega else (phi, psi)
		return (residue, *angles) if include_residue else angles

	for residue in residues:

		# Whether this residue is contiguous with the last and angles calculated
		# from that residue's atoms are valid
		is_contiguous = last_valid and residue.seq == last_residue.seq + 1

		# Reset the generator if not using atoms from last residue
		if not is_contiguous:
			angle_calculator = dihedral_calculator()
			angle_calculator.send(None)  # Prime it

		# Get N, CA, and C atoms from residue
		backbone_atoms = get_backbone_atoms(residue)

		if None in backbone_atoms:
			# Didn't get all backbone atoms - residue is invalid
			is_valid = False
			psi = omega = phi = None

		else:
			# Residue good
			is_valid = True

			# Get backbone atom coords and calculate angles for residue
			backbone_coords = [a.coord for a in backbone_atoms]

			psi = angle_calculator.send(backbone_coords[0])
			omega = angle_calculator.send(backbone_coords[1])
			phi = angle_calculator.send(backbone_coords[2])

		# Yield angles for the previous residue (because calculating psi
		# required an atom from this residue)
		if last_residue is not None:
			yield yield_vals(
				last_residue,
				last_omega if last_contiguous else None,
				last_phi if last_contiguous else None,
				psi if is_contiguous else None,
			)

		# Keep track of state for previous residue
		last_residue = residue
		last_contiguous = is_contiguous
		last_valid = is_valid
		last_omega = omega
		last_phi = phi

	# Last one is only partial - no value for psi
	yield yield_vals(
		last_residue,
		last_omega if last_contiguous else None,
		last_phi if last_contiguous else None,
		None
	)


def torsion_array(residues, include_ends=False, include_omega=False):
	"""Create a Numpy array of phi/psi torsion angles for an amino acid sequence.

	:param residues: Sequence of :class:`.PDBResidue`.
	:param bool include_ends: If True include angles for first and last residues,
		which will contain NaN values. If False omit angles for first and last.
	:parm bool include_omega: Include omega angles in first column of array.

	:returns: Numpy array containing torsion angles for each residue in rows.
		Columns are omega, phi and psi if ``include_omega`` is True or just phi
		and psi otherwise.
	:rtype: np.ndarray
	"""
	import numpy as np

	residues = list(residues)
	reslen = len(residues)

	nrow = reslen if include_ends else reslen - 2
	ncol = 3 if include_omega else 2
	array = np.zeros((nrow, ncol))

	torsion = calc_torsion(residues, include_omega=include_omega)

	# Skip first if needed
	if not include_ends:
		next(torsion)

	# Insert values into array, subbing NaN for None
	for i, angles in zip(range(nrow), torsion):
		for j, a in enumerate(angles):
			array[i, j] = np.nan if a is None else a

	return array
