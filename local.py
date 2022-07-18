from pyscf import lib
from pyscf.dft import numint, gen_grid
import numpy as np

def get_pao(mol, lmo, ortho=True):
    S = mol.intor('int1e_ovlp')
    nao = mol.nao
    P = np.eye(nao) - lib.einsum('mi,ni,nM->mM',lmo, lmo, S)
    if ortho:
        norm = lib.einsum('Mm,mn,Mn->M', P, S, P)
        print('PAO norm ', norm)
        P = lib.einsum('Mm,M->Mm', P, norm**(-1))
    return P

def get_doi(mol, mo):
    nao = mol.nao
    doi_ao = np.zeros((nao, nao))
    ni = numint.NumInt()
    grids = gen_grid.Grids(mol).build()
    for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 0, 2000):
        doi_ao += lib.einsum('gu,gu,gv,gv,g->uv', ao, ao, ao, ao, weight)
    #print(doi_ao[:14,:14])
    #print(doi_ao[:14,14:])
    #print(doi_ao[14:,:14])
    #print(doi_ao[14:,14:])
    doi_mo = lib.einsum('ui,uv,vj->ij', mo, doi_ao, mo)#**0.5
    return doi_mo
