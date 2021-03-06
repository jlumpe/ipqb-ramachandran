#!/usr/bin/env python

"""iPQB boot camp homework assignment.

Author: (Michael) Jared Lumpe

Contains functions for (partially) parsing PDB files, calculating torsion angles
for polypeptides, and making several types of Ramachandran plots.

May be run as a script to parse PDB files and output a plot (run with no
arguments to print usage).

PDB files used as input are assumed to be version 3.3, see
http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
for documentation on the format.
"""

from collections import namedtuple

import math
import argparse
import os
import sys


################################################################################
#
# Parsing and data types
#
################################################################################

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


################################################################################
#
# Dihedrals and torsion calculation
#
################################################################################

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


################################################################################
#
# Plotting
#
################################################################################

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


def format_ramachandran_axes_degrees(ax):
	"""Format matplotlib axes for a ramachandran plot in degrees.

	Sets axis limits, labels, and formats tick labels.

	:param ax: Axes to format.
	:type ax: matplotlib.axes.Axes
	"""

	# Phi and psi labels
	ax.set_xlabel('$\\phi$')
	ax.set_ylabel('$\\psi$')

	# Axis limits
	ax.set_xlim([-180, 180])
	ax.set_ylim([-180, 180])

	# Degree sign in tick labels
	from matplotlib.ticker import StrMethodFormatter
	formatter = StrMethodFormatter('{x}\u00b0')
	ax.xaxis.set_major_formatter(formatter)
	ax.yaxis.set_major_formatter


def format_ramachandran_axes_radians(ax):
	"""Format matplotlib axes for a ramachandran plot in radians.

	Sets axis limits, labels, and formats tick labels.

	Axis tick labels will include a pi symbol, but actual data values won't be
	multiples of pi. So data should be scaled by 1/pi before plotting.

	:param ax: Axes to format.
	:type ax: matplotlib.axes.Axes
	"""

	# Phi and psi labels
	ax.set_xlabel('$\\phi$')
	ax.set_ylabel('$\\psi$')

	# Axis limits
	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])

	# Pi in tick labels
	from matplotlib.ticker import StrMethodFormatter
	formatter = StrMethodFormatter('{x}\u03c0')
	ax.xaxis.set_major_formatter(formatter)
	ax.yaxis.set_major_formatter(formatter)


def get_ramachandran_xy(data, scale=None):
	"""Convert torsion angle data into a consistent format for plotting.

	:param data: Phi and psi angles either as a list of 2-tuples or a numpy
		array with pairs of angles in rows.
	:param str scale: Scale the data. Valid values are "degrees" to convert
		from radians to degrees and "overpi" to divide by pi in order to use
		radians for the axis ticks.
	:returns: ``(x, y)`` (``(phi, psi)``) tuple of 1D numpy arrays.
	:rtype: tuple[numpy.ndarray]
	"""

	import numpy as np

	if isinstance(data, np.ndarray):
		x, y = data.T

	else:
		x, y = map(np.asarray, zip(*data))

	if scale == 'degrees':
		x, y = map(np.rad2deg, (x, y))

	elif scale == 'overpi':
		x, y = x / np.pi, y / np.pi

	elif scale is not None:
		raise ValueError(scale)

	return x, y


def ramachandran_hexbin(data, ax=None, degrees=True, log=False, **kwargs):
	"""Ramachandran plot using :func:`matplotlib.pyplot.hexbin`.

	:param data: Phi and psi angles either as a list of 2-tuples or a numpy
		array with pairs of angles in rows.
	:param ax: Axis to plot on. If None will use current axis.
	:type ax: matplotlib.axes.Axes
	:param bool degrees: Format axes for degrees (True) or radians (False).
	:param bool log: Color bins according to log-density.
	:param \\**kwargs: Additional keyword arguments to
		:func:`matplotlib.pyplot.hexbin`.

	:returns: Return value of the ``hexbin()`` function.
	:rtype: matplotlib.collections.PolyCollection
	"""

	import matplotlib.pyplot as plt

	# Default arguments to hexbin()
	kwargs.setdefault('cmap', 'magma')
	if log:
		kwargs.setdefault('bins', 'log')

	# Get scaled x/y values
	x, y = get_ramachandran_xy(data, 'degrees' if degrees else 'overpi')

	# Use current axis by default
	if ax is None:
		ax = plt.gca()

	# Plot hexbin
	extent = (-180, 180, -180, 180) if degrees else (-1, 1, -1, 1)
	hb = ax.hexbin(x, y, extent=extent, **kwargs)

	# Format axes
	if degrees:
		format_ramachandran_axes_degrees(ax)
	else:
		format_ramachandran_axes_radians(ax)

	return hb


def ramachandran_scatter(data, ax=None, degrees=True, **kwargs):
	"""Ramachandran plot using :func:`matplotlib.pyplot.scatter`.

	:param data: Phi and psi angles either as a list of 2-tuples or a numpy
		array with pairs of angles in rows.
	:param ax: Axis to plot on. If None will use current axis.
	:type ax: matplotlib.axes.Axes
	:param bool degrees: Format axes for degrees (True) or radians (False).
	:param \\**kwargs: Additional keyword arguments to
		:func:`matplotlib.pyplot.scatter`.

	:returns: Return value of the ``scatter()`` function.
	:rtype: matplotlib.collections.PathCollection
	"""

	import matplotlib.pyplot as plt

	# Get scaled x/y values
	x, y = get_ramachandran_xy(data, 'degrees' if degrees else 'overpi')

	# Use current axis by default
	if ax is None:
		ax = plt.gca()

	# Plot scatter
	plot = ax.scatter(x, y, **kwargs)

	# Format axes
	if degrees:
		format_ramachandran_axes_degrees(ax)
	else:
		format_ramachandran_axes_radians(ax)

	return plot


def ramachandran_jointplot(data, degrees=True, **kwargs):
	"""Ramachandran plot using :func:`seaborn.jointplot`.

	Requires the seaborn package.

	:param data: Phi and psi angles either as a list of 2-tuples or a numpy
		array with pairs of angles in rows.
	:param bool degrees: Format axes for degrees (True) or radians (False).
	:param \\**kwargs: Additional keyword arguments to :func:`seaborn.jointplot`.

	:returns: Return value of the ``jpointplot()`` function.
	:rtype: seaborn.axisgrid.JointGrid
	"""

	from seaborn import jointplot

	# Default arguments to jointplot()
	kwargs.setdefault('stat_func', None)

	# Get scaled x/y values
	x, y = get_ramachandran_xy(data, 'degrees' if degrees else 'overpi')

	# Plot jointplot
	lim = (-180, 180) if degrees else (-1, 1)
	plot = jointplot(x, y, xlim=lim, ylim=lim, **kwargs)

	# Format axes
	if degrees:
		format_ramachandran_axes_degrees(plot.ax_joint)
	else:
		format_ramachandran_axes_radians(plot.ax_joint)

	return plot


def ramachandran_kdeplot(data, degrees=True, **kwargs):
	"""Ramachandran plot using :func:`seaborn.kdeplot`.

	Requires the seaborn package.

	:param data: Phi and psi angles either as a list of 2-tuples or a numpy
		array with pairs of angles in rows.
	:param bool degrees: Format axes for degrees (True) or radians (False).
	:param \\**kwargs: Additional keyword arguments to
		:func:`seaborn.kdeplot`.

	:returns: Return value of the ``kdeplot()`` function.
	:rtype: matplotlib.axes.Axes
	"""

	from seaborn import kdeplot

	# Get scaled x/y values
	x, y = get_ramachandran_xy(data, 'degrees' if degrees else 'overpi')

	# Plot jointplot
	ax = kdeplot(x, y, **kwargs)

	# Format axes
	if degrees:
		format_ramachandran_axes_degrees(ax)
	else:
		format_ramachandran_axes_radians(ax)

	return ax


################################################################################
#
# Command line interface
#
################################################################################

cli = argparse.ArgumentParser(
	description='Parse PDB files and create a Ramachandran plot.'
)
cli.add_argument('files', nargs='+', type=str,
                 help='PDB files to parse')
cli.add_argument('-t', '--type', dest='plottype', type=str, default='hexbin',
                 choices=['hexbin', 'scatter', 'kde', 'joint'],
                 help='Type of plot to produce')
cli.add_argument('-r', '--radians', action='store_true',
                 help='Label axes with radians instead of degrees')
cli.add_argument('--log', action='store_true',
                 help='Use log color scale for hexbin plot')
cli.add_argument('--cbar', action='store_true',
                 help='Display color bar for hexbin plot')


def _exit(message):
	"""Print a message to stderr and exit with non-zero exit code."""
	print('ERROR: ' + message, file=sys.stderr)
	sys.exit(1)


def _assert_seaborn_available(plottype):
	"""Print error message and quit if seaborn not available."""
	try:
		import seaborn

	except ImportError:
		_exit(
			'The seaborn package must be installed to create {} plots'
			.format(plottype)
		)


def main():
	"""Execute command line interface."""

	import numpy as np
	import matplotlib as mpl

	# Parse arguments
	args = cli.parse_args()

	# Check all files exist
	for filepath in args.files:
		if not os.path.isfile(filepath):
			_exit('file {} does not exist'.format(filepath))

	# Use TK interactive backend as it should work on all installations
	mpl.use('tkagg')

	# Turn on interactive plotting (pyplot import must come after call to use())
	import matplotlib.pyplot as plt
	plt.ion()

	# Get torsion angle arrays for all input files
	arrays = []
	for filepath in args.files:
		with open(filepath) as fobj:
			chain = list(parse_pdb_chain(fobj))
			arrays.append(torsion_array(chain))

	angles = np.concatenate(arrays, axis=0)

	# Make the plot
	if args.plottype == 'hexbin':
		hb = ramachandran_hexbin(angles, log=args.log, degrees=not args.radians)
		if args.cbar:
			plt.colorbar(hb)

	elif args.plottype == 'scatter':
		ramachandran_scatter(angles, degrees=not args.radians)

	elif args.plottype == 'kde':
		_assert_seaborn_available('kde')
		ramachandran_kdeplot(angles, shade=True, degrees=not args.radians)

	elif args.plottype == 'joint':
		_assert_seaborn_available('joint')
		ramachandran_jointplot(angles, kind='kde', degrees=not args.radians)

	else:
		raise ValueError(args.plottype)

	# Plot title
	if len(args.files) == 1:
		filename = os.path.basename(args.files[0])
	else:
		filename = '{} files'.format(len(args.files))

	plt.gcf().suptitle('{} ({} residues)'.format(filename, angles.shape[0]))

	# Show the plot, blocking until closed
	plt.show(block=True)


# Call main func if run as script
if __name__ == '__main__':
	main()
