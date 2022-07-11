from pyscf import gto, lib
from pyscf.gto.mole import PTR_ENV_START #_rm_digit, NUC_POINT, ATOM_OF, ATM_SLOTS, BAS_SLOTS, _parse_nuc_mod, make_atm_env, make_bas_env
from pyscf.gto.moleintor import getints
import sys
import numpy as np

def nao_by_atom(mol):
    nao_lst = []
    for a, (s1, s2, b1, b2) in enumerate(mol.aoslice_by_atom()):
        nao_lst.append((s2-s1,b2-b1))
    return nao_lst

def make_env_IJ(mol):
    #print(mol._atom)
    #print(mol._basis)
    #nao_lst = nao_by_atom(mol)
    aoslice = mol.aoslice_by_atom()
    abe = []
    for I in range(mol.natm):
        fake_atom = [mol._atom[I]]
        fake_basis = mol._basis
        _atm, _bas, _env = gto.mole.make_env(fake_atom, fake_basis, 
                    np.zeros(PTR_ENV_START), mol.nucmod, mol.nucprop)
        abe.append([_atm, _bas, _env, aoslice[I], None])

    for I in range(mol.natm):
        for J in range(I+1, mol.natm):
            fake_atom = [mol._atom[I], mol._atom[J]]
            fake_basis = mol._basis
            _atm, _bas, _env = gto.mole.make_env(fake_atom, fake_basis, 
                    np.zeros(PTR_ENV_START), mol.nucmod, mol.nucprop)
            #print(_atm)
            #print(_bas)
            #print(_env)
            abe.append([_atm, _bas, _env, aoslice[I], aoslice[J]])
    return abe


def riints(mol, auxmol):
    abe = make_env_IJ(mol)
    abe_aux = make_env_IJ(auxmol)
    
    intor3c2e = mol._add_suffix('int3c2e')
    intor2c2e = mol._add_suffix('int2c2e')
    #hermi = 0
    #ao_loc = None
    #aoslice = mol.aoslice_by_atom()
    #aoslice_aux = auxmol.aoslice_by_atom()
    ints_IJ = []
    for I in range(mol.natm):
        atm_I, bas_I, env_I, aosl_I, _ = abe[I]
        atmx_I, basx_I, envx_I, _, _ = abe_aux[I]
        atm, bas, env = gto.mole.conc_env(atm_I, bas_I, env_I,
                                      atmx_I, basx_I, envx_I)
        nbas = bas_I.shape[0]
        nauxbas = basx_I.shape[0]
        shls_slice = (0, nbas, 0, nbas, nbas, nbas+nauxbas)
        ints = getints(intor3c2e, atm, bas, env, shls_slice)
        V = getints(intor2c2e, atmx_I, basx_I, envx_I, (0, nauxbas, 0, nauxbas))
        C = lib.einsum('ijP, PQ -> ijQ', ints, np.linalg.inv(V))
        aosl = slice(aosl_I[2], aosl_I[3])
        ints_IJ.append([C, aosl, aosl])
        print('Pair %d'%I, 'RI ints shape', ints.shape, 'V shape', V.shape)
    for IJ in range(mol.natm, len(abe)):
        atm_IJ, bas_IJ, env_IJ, aosl_I, aosl_J = abe[IJ]
        atmx_IJ, basx_IJ, envx_IJ, _, _ = abe_aux[IJ]
        atm, bas, env = gto.mole.conc_env(atm_IJ, bas_IJ, env_IJ,
                                      atmx_IJ, basx_IJ, envx_IJ)
        nbas = bas_IJ.shape[0]
        nauxbas = basx_IJ.shape[0]
        nshl_I = aosl_I[1] - aosl_I[0]
        shls_slice = (0, nshl_I, nshl_I, nbas, nbas, nbas+nauxbas)
        ints = getints(intor3c2e, atm, bas, env, shls_slice)
        V = getints(intor2c2e, atmx_IJ, basx_IJ, envx_IJ, (0, nauxbas, 0, nauxbas))
        C = lib.einsum('ijP, PQ -> ijQ', ints, np.linalg.inv(V))
        aosls_I = slice(aosl_I[2], aosl_I[3])
        aosls_J = slice(aosl_J[2], aosl_J[3])
        ints_IJ.append([C, aosls_I, aosls_J])
        print('Pair %d'%IJ, 'RI ints shape', ints.shape, 'V shape', V.shape)
    return ints_IJ

def riints2(auxmol):
    abe_aux = make_env_IJ(auxmol)
    intor2c2e = auxmol._add_suffix('int2c2e')
    V2 = []
    for IJ in range(len(abe_aux)):
        V2_IJ = []
        atmx_IJ, basx_IJ, envx_IJ, _, _ = abe_aux[IJ]
        nbas_IJ = basx_IJ.shape[0]
        #for KL in range(IJ, len(abe_aux)):
        for KL in range(len(abe_aux)):
            atmx_KL, basx_KL, envx_KL, _, _ = abe_aux[KL]
            nbas_KL = basx_KL.shape[0]
            atm, bas, env = gto.mole.conc_env(atmx_IJ, basx_IJ, envx_IJ,
                                      atmx_KL, basx_KL, envx_KL)
            shls_slice = (0, nbas_IJ, nbas_IJ, nbas_IJ + nbas_KL)
            ints = getints(intor2c2e, atm, bas, env, shls_slice)
            print('Pair %d-%d'%(IJ,KL), '(P|Q) ints shape', ints.shape)
            V2_IJ.append(ints)
        V2.append(V2_IJ)
    return V2

def get_eri(mol, auxmol):
    ints_3c = riints(mol, auxmol)
    ints_V = riints2(auxmol)
    nao = nmo = mol.nao
    eri = np.zeros((nao, nao, nao, nao))
    size = 0
    for IJ in range(len(ints_3c)):
        int_3c_IJ, aosl1, aosl2 = ints_3c[IJ]
        #if IJ < mol.natm:
        #    fac1 = 1.0
        #else:
        #    fac1 = 2.0
        #for KL in range(IJ, len(ints_3c)):
        for KL in range(len(ints_3c)):
            #print('%d-%d'%(IJ,KL))
            int_3c_KL, aosl3, aosl4 = ints_3c[KL]
            int_V = ints_V[IJ][KL]
            #print(int_3c_IJ.shape, int_V.shape, int_3c_KL.shape)
            #if KL < mol.natm:
            #    fac2 = 1.0
            #else:
            #    fac2 = 2.0
            ints = lib.einsum("uvP, PQ, klQ -> uvkl", int_3c_IJ, int_V, int_3c_KL)
            eri[aosl1, aosl2, aosl3, aosl4] = ints
            print(IJ, KL, aosl1, aosl2, aosl3, aosl4)
            s = ints.size
            size += s
            if aosl1 != aosl2:
                eri[aosl2, aosl1, aosl3, aosl4] = ints.transpose(1,0,2,3)
                size += s
            if aosl3 != aosl4:
                eri[aosl1, aosl2, aosl4, aosl3] = ints.transpose(0,1,3,2)
                #print('0132')
                size += s
            if aosl1 != aosl2 and aosl3 != aosl4:
                eri[aosl2, aosl1, aosl4, aosl3] = ints.transpose(1,0,3,2)
                #print('1032')
                size += s
    print(size, nao**4)
    return eri


def rimp2(mol, auxmol, mo_coeff):
    ints_3c = riints(mol, auxmol)
    ints_V = riints2(auxmol)
    nao = nmo = mol.nao
    nocc = mol.nelec[0]
    nvir = nmo - nocc
    so, sv, sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
    Co, Cv = mo_coeff[:, so], mo_coeff[:, sv]
    int_iajb = np.zeros((nocc, nvir, nocc, nvir))
    for IJ in range(len(ints_3c)):
        int_3c_IJ, aosl1, aosl2 = ints_3c[IJ]
        if IJ < mol.natm:
            fac1 = 1.0
        else:
            fac1 = 2.0
        #for KL in range(IJ, len(ints_3c)):
        for KL in range(len(ints_3c)):
            print('%d-%d'%(IJ,KL))
            int_3c_KL, aosl3, aosl4 = ints_3c[KL]
            int_V = ints_V[IJ][KL]
            #print(int_3c_IJ.shape, int_V.shape, int_3c_KL.shape)
            if KL < mol.natm:
                fac2 = 1.0
            else:
                fac2 = 2.0
            int_iajb += fac1*fac2*lib.einsum("ui, va, uvP, PQ, klQ, kj, lb -> iajb", 
                    Co[aosl1,:], Cv[aosl2,:], int_3c_IJ, int_V, int_3c_KL, 
                    Co[aosl3,:], Cv[aosl4,:])
    return int_iajb
