from pyscf import gto, scf
import rilvl

mol = gto.Mole()
mol.atom = """
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.0  0.0  0.0
H  0.0  0.7  1.0
"""
mol.basis = "def2-tzvp"
mol.verbose = 0
mol.build()
print(mol.aoslice_by_atom())

auxmol = mol.copy()
auxmol.basis = "def2-tzvp-jkfit"
auxmol.build()
print(auxmol.aoslice_by_atom())

#rilvl.make_env_IJ(mol)
rilvl.riints(mol, auxmol)
