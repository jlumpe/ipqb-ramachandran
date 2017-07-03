"""Global test configuration and fixtures."""

import pytest


@pytest.fixture(scope='module')
def pdbfile():
	"""Path to a PDB file to parse. You'll need to actually put it there."""
	return 'structures/12asA00'
