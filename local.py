from pyscf import lib
from pyscf.dft import numint, gen_grid
from pyscf.lo import orth
import numpy as np
import scipy
import rilvl
from timing import timing

def get_pao(mol, lmo, ortho=True):
    S = mol.intor('int1e_ovlp')
    nao = mol.nao
    P = np.eye(nao) - lib.einsum('mi,ni,nM->mM',lmo, lmo, S)
    if ortho:
        norm = lib.einsum('mM,mn,nM->M', P, S, P)**0.5
        print('PAO norm ', norm)
        P = lib.einsum('mM,M->mM', P, norm**(-1))
        ovlp = lib.einsum('mM,mn,nN->MN', P, S, P)
        #print(ovlp)
    return P

def quasi_can(pao, fock_ao):
    fock_pao = lib.einsum('ma, mn, nb->ab', pao, fock_ao, pao)
    #print('PAO fock diag', fock_pao.diagonal())
    evir, v = scipy.linalg.eigh(fock_pao)
    qc_pao = lib.einsum('ma,aA->mA', pao, v.T)
    
    return evir, qc_pao

def mo_fock(mo, fock_ao):
    fock_lmo = lib.einsum('ma, mn, nb->ab', mo, fock_ao, mo)
    eocc = fock_lmo.diagonal()
    return eocc


def get_pno(mol, lmo, pao, doi_l, doi_lp, fock_ao, auxmol):
    nlmo = lmo.shape[1]
    print(doi_lp.shape)
    eocc = mo_fock(lmo, fock_ao)
    domain_info = {}
    for i in range(nlmo):
        for j in range(i,nlmo):
            if doi_l[i,j] < 1e-4:
                print('skip %d,%d doi %.6f' %(i,j,doi_l[i,j]))
                continue
            link_ip = list(np.flatnonzero(doi_lp[i,:]>1e-4))
            link_jp = list(np.flatnonzero(doi_lp[j,:]>1e-4))
            link_ijp = list(set(link_ip + link_jp))
            print(i,j, len(link_ip), len(link_jp), len(link_ijp))
            pao_ij = pao[:, link_ijp]
            evir, qc_pao_ij = quasi_can(pao, fock_ao)
            M_atom, M_loc, M_loc_ao = pop_mo(mol, qc_pao_ij, thresh=1e-2)
            domain_info[(i,j)] = pao_ij, M_loc
    ints_3c = rilvl.riints(mol, auxmol)
    ints_V = rilvl.riints2(auxmol)
    for IJ in range(len(ints_3c)):

    



def check_ortho(mol, mo):
    S = mol.intor('int1e_ovlp')
    ovlp = lib.einsum('mM,mn,nN->MN', mo, S, mo)
    #print(ovlp)
    ort_idx = np.linalg.norm(ovlp - np.eye(ovlp.shape[0]))
    print('Ortho index', ort_idx)


@timing
def get_doi(mol, mo):
    nao = mol.nao
    doi_ao = np.zeros((nao, nao, nao, nao))
    ni = numint.NumInt()
    grids = gen_grid.Grids(mol).build()
    for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 0, 2000):
        doi_ao += lib.einsum('gu,gv,gk,gl,g->uvkl', ao, ao, ao, ao, weight)
    #print(doi_ao[:14,:14])
    #print(doi_ao[:14,14:])
    #print(doi_ao[14:,:14])
    #print(doi_ao[14:,14:])
    doi_mo = lib.einsum('ui, vi, uvkl, kj, lj->ij', mo,mo, doi_ao, mo,mo)#**0.5
    return doi_mo

def pop_mo(mol, mo, lowdin=False, thresh=1e-3):
    S = mol.intor('int1e_ovlp')
    print(S)
    if lowdin:
        orth_coeff = orth.orth_ao(mol, 'meta_lowdin', 'ANO', s=S)
        c_inv = np.dot(orth_coeff.conj().T, S)
        mo = np.dot(c_inv, mo)
    M = lib.einsum('mi,mn,ni->im', mo, S, mo)
    #print(M[:,:20])
    M_atom = np.zeros((M.shape[0], mol.natm))
    for a, (s1, s2, b1, b2) in enumerate(mol.aoslice_by_atom()):
        M_atom[:,a] = np.sum(M[:,slice(b1,b2)], axis=1)
    #print(M_atom)
    M_loc = {}
    M_loc_ao = {}
    aoslice = mol.aoslice_by_atom()
    for i in range(M.shape[0]):
        atmloc = np.flatnonzero(abs(M_atom[i])>thresh)
        M_loc[i] = atmloc
        loc_ao = []
        for A in atmloc:
            loc_ao += list(range(aoslice[A][2], aoslice[A][3]))
        M_loc_ao[i] = loc_ao
    print(M_loc)
    return M_atom, M_loc, M_loc_ao

@timing
def get_doi_local(mol, mo1, mo2):
    M1_atom, M1_loc, M1_loc_ao = pop_mo(mol, mo1)
    M2_atom, M2_loc, M2_loc_ao = pop_mo(mol, mo2)
    nmo1 = mo1.shape[1]
    nmo2 = mo2.shape[1]
    doi_mo = np.zeros((nmo1, nmo2))
    nao = mol.nao
    ni = numint.NumInt()
    grids = gen_grid.Grids(mol).build()
    for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 0, 2000):
        for i in range(nmo1):
            iloc = M1_loc_ao[i]
            doi_tmpi = lib.einsum('u, gu', mo1[iloc,i], ao[:,iloc])**2
            for j in range(nmo2):
                jloc = M2_loc_ao[j]
                doi_tmpj = lib.einsum('u, gu', mo2[jloc,j], ao[:,jloc])**2
                doi_mo[i,j] += lib.einsum('g,g,g->', doi_tmpi, doi_tmpj, weight)
    return np.sqrt(doi_mo)

