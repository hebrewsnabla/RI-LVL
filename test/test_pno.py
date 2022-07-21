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
H  1.0  0.0  0.0
H -0.26 0.96 0.0
O  0.0 -3.0  1.0
H  1.0 -3.0  1.0
H -0.26 -3.96 1.0
"""
mol.basis = "cc-pvdz"
mol.verbose = 0
mol.build()
print(mol.aoslice_by_atom())

mf = mol.RHF().run()

auxmol = mol.copy()
auxmol.basis = "cc-pvdz-ri"
auxmol.build()

pm = lo.PM(mol, mf.mo_coeff[:,mf.mo_occ>0], mf)
lmo = pm.kernel()
dump_mat.dump_mo(mol, lmo, ncol=12)
#pop = pm.atomic_pops(mol, lmo)
#print(pop)
doi_l =local.get_doi_local(mol,lmo, lmo)
print(doi_l)
#doi = local.get_doi(mol,lmo)
#print(doi)
#exit()
pao = local.get_pao(mol,lmo)
dump_mat.dump_mo(mol, pao, ncol=12)
"""evir, eocc, qc_pao = local.quasi_can(pao, lmo, mf.get_fock())
print(evir)
print(eocc)
#dump_mat.dump_mo(mol, qc_pao, ncol=12)
local.check_ortho(mol, qc_pao)
"""
doi_lp =local.get_doi_local(mol,lmo, pao)
#print(doi_lp)

local.get_pno(mol, lmo, pao, doi_l, doi_lp, mf.get_fock())
