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
PDBAtom = namedtuple('PDBAtom', [field for b, e, t, field in PDB_ATOM_LINE_FIELDS])


# Namde tuple containing parsed data on a PDB residue and its atoms
PDBResidue = namedtuple('PDBResidue', 'resName,chainID,resSeq,atoms')


def parse_pdb_atom(line):
	"""Parse an ATOM line from a PDB file.

	:type line: str
	:rtype: .PDBAtom
	"""
	elems = []

	for begin, end, type_, fieldname in PDB_ATOM_LINE_FIELDS:
		strval = line[begin:end].strip()

		if strval:
			val = type_(strval)
		else:
			val = None

		elems.append(val)

	return PDBAtom(*elems)


def parse_pdb_chain(fobj):
	"""Parse a chain of residues from a PDB file.

	File must contain only a single chain.

	This is able to parse all ~12k PDB files in the link provided in the
	bootcamp website (no errors, but didn't actually validate output).

	:param fobj: Open file object in text mode. Alternatively, may be any
		iterable of PDB file lines.
	:returns: List of :class:`.PDBResidue`.
	"""

	residues = []
	currentres = None

	for line in fobj:
		# Ignore non-ATOM lines
		if not line.startswith('ATOM'):
			continue

		# Parse line
		atom = parse_pdb_atom(line.strip())

		# Create a new residue if needed
		if currentres is None or atom.resSeq != currentres.resSeq:

			# Check same chain
			if currentres is not None and atom.chainID != currentres.chainID:
				raise ValueError('PDB files with multiple chains not supported.')

			currentres = PDBResidue(atom.resName, atom.chainID, atom.resSeq, [])
			residues.append(currentres)

		# Add atom to residue
		currentres.atoms.append(atom)

	return residues
