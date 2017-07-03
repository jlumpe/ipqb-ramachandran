"""Test PDB file parsing."""

import pytest
import numpy as np
from Bio import PDB

from ramachandran import parse_pdb_chain


def test_parse_chain(pdbfile):
	"""Test parsing residues from a PDB file vs the BioPython implementation."""

	# Parse using our code
	with open(pdbfile) as fobj:
		residues1 = list(parse_pdb_chain(fobj))

	# Parse using BioPython
	parser = PDB.PDBParser()
	structure = parser.get_structure('test', pdbfile)
	residues2 = list(structure.get_residues())

	assert len(residues1) == len(residues2)

	# Compare residues
	for res1, res2 in zip(residues1, residues2):

		# Residue attributes
		assert res1.name == res2.resname
		assert res1.seq == res2.id[1]

		# Compare atoms
		assert len(res1.atoms) == len(res2)

		# Both should be in the same order they were in in the file...
		for a1, a2 in zip(res1.atoms, res2):
			assert a1.name == a2.name
			assert np.allclose(a1.coord, a2.coord)
			assert a1.serial == a2.serial_number
