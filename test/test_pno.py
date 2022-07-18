import sys
sys.path.append("..")
from pyscf import gto, lo
from automr import dump_mat
import local
import numpy as np
np.set_printoptions(precision=6, linewidth=160, suppress=True)

mol = gto.Mole()
mol.atom = """
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.0  0.0  0.0
H  0.0  0.7  1.0
"""
mol.basis = "cc-pvdz"
mol.verbose = 0
mol.build()
print(mol.aoslice_by_atom())

mf = mol.RHF().run()

pm = lo.PM(mol, mf.mo_coeff[:,mf.mo_occ>0], mf)
lmo = pm.kernel()
dump_mat.dump_mo(mol, lmo, ncol=10)
doi = local.get_doi(mol,lmo)
print(doi)
pao = local.get_pao(mol,lmo)
dump_mat.dump_mo(mol, pao, ncol=10)
