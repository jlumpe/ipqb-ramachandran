"""Test calculation of dihedral and torsion angles."""

import pytest

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Vector import Vector, calc_dihedral as biopy_calc_dihedral

import ramachandran as ram


@pytest.fixture
def residues(pdbfile):
	"""List of residues parsed from PDB file."""
	with open(pdbfile) as fobj:
		return list(ram.parse_pdb_chain(fobj))


@pytest.fixture
def polypeptide(pdbfile):
	parser = PDBParser()
	structure = parser.get_structure('test', pdbfile)

	builder = PPBuilder()
	pp, = builder.build_peptides(structure)

	return pp


def test_calc_dihedrals(residues):
	"""Test the calc_dihedrals function vs BioPython's version."""

	backbone = [a for res in residues for a in ram.get_backbone_atoms(res)]

	for i in range(1, len(residues) - 1):
		assert residues[i].seq == residues[i - 1].seq + 1

	dihedrals = list(ram.calc_dihedrals(atom.coord for atom in backbone))

	for i, angle1 in enumerate(dihedrals):
		vectors = [Vector(atom.coord) for atom in backbone[i:i + 4]]
		angle2 = biopy_calc_dihedral(*vectors)

		assert np.isclose(angle1, angle2)


@pytest.mark.parametrize('calc_omega', [False, True])
def test_calc_torsion(residues, polypeptide, calc_omega):
	"""Test the calc_torsion function vs BioPython's version."""

	torsion = [angles for res, *angles in ram.calc_torsion(residues)]
	biopy_torsion = polypeptide.get_phi_psi_list()

	assert len(torsion) == len(biopy_torsion)

	# Apparently there is some significant numerical error in one of the two
	# methods, so we need to use a lower tolerance than the default
	def close(x, y):
		if x is None or y is None:
			return x is None and y is None
		else:
			return np.isclose(x, y, rtol=1e-3)

	for angles1, (phi2, psi2) in zip(torsion, biopy_torsion):
		phi1, psi1 = angles1[-2:]

		assert close(phi1, phi2)
		assert close(psi1, psi2)


@pytest.mark.parametrize('include_ends', [False, True])
@pytest.mark.parametrize('include_omega', [False, True])
def test_torsion_array(residues, include_ends, include_omega):
	"""Test the calc_torsion function vs BioPython's version."""

	torsion_list = [
		angles for res, *angles in
		ram.calc_torsion(residues, include_omega=include_omega)
	]
	torsion_array = ram.torsion_array(residues, include_ends=include_ends,
	                                  include_omega=include_omega)

	list_compare = torsion_list if include_ends else torsion_list[1:-1]

	assert len(list_compare) == torsion_array.shape[0]

	for vals1, vals2 in zip(list_compare, torsion_array):
		assert len(vals1) == len(vals2)

		for v1, v2 in zip(vals1, vals2):
			if v1 is None:
				assert np.isnan(v2)
			else:
				assert v1 == v2
