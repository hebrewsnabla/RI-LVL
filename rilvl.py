from pyscf import gto
from pyscf.gto.mole import PTR_ENV_START #_rm_digit, NUC_POINT, ATOM_OF, ATM_SLOTS, BAS_SLOTS, _parse_nuc_mod, make_atm_env, make_bas_env
from pyscf.gto.moleintor import getints
import sys
import numpy as np

def make_env_IJ(mol):
    #print(mol._atom)
    #print(mol._basis)
    abe = []
    for I in range(mol.natm):
        for J in range(I+1, mol.natm):
            fake_atom = [mol._atom[I], mol._atom[J]]
            fake_basis = mol._basis
            _atm, _bas, _env = gto.mole.make_env(fake_atom, fake_basis, 
                    np.zeros(PTR_ENV_START), mol.nucmod, mol.nucprop)
            #print(_atm)
            #print(_bas)
            #print(_env)
            abe.append([_atm, _bas, _env])
    return abe


def riints(mol, auxmol):
    abe = make_env_IJ(mol)
    abe_aux = make_env_IJ(auxmol)
    
    intor = mol._add_suffix('int3c2e')
    #hermi = 0
    #ao_loc = None
    #aoslice = mol.aoslice_by_atom()
    #aoslice_aux = auxmol.aoslice_by_atom()
    for IJ in range(len(abe)):
        atm_IJ, bas_IJ, env_IJ = abe[IJ]
        atmx_IJ, basx_IJ, envx_IJ = abe_aux[IJ]
        atm, bas, env = gto.mole.conc_env(atm_IJ, bas_IJ, env_IJ,
                                      atmx_IJ, basx_IJ, envx_IJ)
        nbas = bas_IJ.shape[0]
        nauxbas = basx_IJ.shape[0]
        shls_slice = (0, nbas, 0, nbas, nbas, nbas+nauxbas)
        ints = getints(intor, atm, bas, env, shls_slice)
        print('Pair %d'%IJ, 'RI ints shape', ints.shape)
