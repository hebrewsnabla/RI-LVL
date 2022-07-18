from pyscf import gto, scf, mp, lib, df
import rilvl
import scipy
import numpy as np
lib.num_threads(1)

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
#print(auxmol.aoslice_by_atom())

scf_df = scf.RHF(mol).density_fit().run()
mo_coeff = scf_df.mo_coeff
#mp2_df = mp.MP2(scf_df).run()

nao = nmo = mol.nao
nocc = mol.nelec[0]
nvir = nmo - nocc
so, sv, sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
Co, Cv = mo_coeff[:, so], mo_coeff[:, sv]
int2c2e = auxmol.intor("int2c2e")
int3c2e = df.incore.aux_e2(mol, auxmol)
nao_df = auxmol.nao
#int2c2e_half = scipy.linalg.cholesky(int2c2e, lower=True)
#V_df_mp2 = scipy.linalg.solve_triangular(int2c2e_half, int3c2e.reshape(-1, nao_df).T, lower=True)\
#               .reshape(nao_df, nao, nao).transpose((1, 2, 0))
#eri_iajb = lib.einsum("ui, va, uvP, klP, kj, lb -> iajb", Co, Cv, V_df_mp2, V_df_mp2, Co, Cv)

#V_df = tril_to_symm(scf_df.with_df._cderi).transpose((1, 2, 0))
df_eri = lib.einsum("uvP, PQ, klQ -> uvkl", int3c2e, np.linalg.inv(int2c2e), int3c2e)
eri = mol.intor("int2e")
#rilvl.riints2(auxmol)
#int_iajb = rilvl.rimp2(mol, auxmol, mo_coeff)
#print(eri_iajb[0,0,0,0], int_iajb[0,0,0,0])
auxmol = mol.copy()
auxmol.basis = "def2-tzvp-jkfit"
auxmol.build()
lvl_eri = rilvl.get_eri(mol, auxmol)
print(np.allclose(lvl_eri, eri, 0.001))
print((lvl_eri-eri).max())

print('i, j, k, l, RI-LVL, RI-V')
err2=0
err3=0
for i in range(nao):
    for j in range(nao):
        for k in range(nao):
            for l in range(nao):
                dd = lvl_eri[i,j,k,l] - eri[i,j,k,l]
                if abs(dd) > 1e-2:
                    #print(i,j,k,l,eri[i,j,k,l], df_eri[i,j,k,l])
                    err2+=1
                if abs(dd) > 1e-3:
                    err3 += 1
print('err num', err2, err3)

import matplotlib.pyplot as plt
bins =np.arange(-12, 1., 0.1)
histo_eri = np.histogram(np.log10(np.abs(eri).ravel() + 1e-20), bins=bins)
histo_df_eri = np.histogram(np.log10(np.abs(df_eri).ravel() + 1e-20), bins=bins)
histo_lvl_eri = np.histogram(np.log10(np.abs(lvl_eri).ravel() + 1e-20), bins=bins)
histo_dev0 = np.histogram(np.log10(np.abs(df_eri - eri).ravel() + 1e-20), bins=bins)
histo_dev = np.histogram(np.log10(np.abs(lvl_eri - eri).ravel() + 1e-20), bins=bins)

prob_eri = histo_eri[0] / eri.size
prob_df_eri = histo_df_eri[0] / eri.size
prob_lvl_eri = histo_lvl_eri[0] / eri.size
prob_dev0 = histo_dev0[0] / eri.size
prob_dev = histo_dev[0] / eri.size

fig, ax = plt.subplots()

ax.plot(histo_eri[1][:-1], prob_eri, label="noRI AO ERI", linestyle="-.")
ax.plot(histo_df_eri[1][:-1], prob_df_eri, label="RI-V AO ERI", linestyle=":")
ax.plot(histo_df_eri[1][:-1], prob_lvl_eri, label="RI-LVL AO ERI", linestyle=":")
ax.plot(histo_dev0[1][:-1], prob_dev0, label="Deviation RI-V vs noRI")
ax.plot(histo_dev[1][:-1], prob_dev, label="Deviation RI-LVL vs noRI")

ax.set_xlabel("$\mathrm{log}_{10} \mathrm{AO\, ERI}$")
ax.set_ylabel("Probability Distribution")
ax.set_title("AO ERI Distribution Status")
ax.legend()
fig.savefig('test.png')
