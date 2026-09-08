import numpy as np
import os
import pickle

def generateTopologyFromSAW(S, fileTag, mainFolder):
    K = 1500
    chainPos  = S['chainPos']
    meta      = S['meta']

    # Direct access to Python dict - no conversion needed!
    no_chains = meta['no_chains']
    no_beads  = meta['no_beads']
    L         = meta['L_grid']

    Natom  = no_chains * no_beads
    Nbonds = no_chains * (no_beads - 1)

    xlo = 0; xhi = L
    ylo = 0; yhi = L
    zlo = 0; zhi = L
    
    xyz   = np.zeros((Natom, 3))
    chain = np.zeros(Natom, dtype=int)
    atype = 2 * np.ones(Natom, dtype=int)
    id0 = 0
    
    for c in range(no_chains):
        P = chainPos[c]
        
        if P.shape[0] != no_beads:
            raise ValueError(f'Chain {c+1} has {P.shape[0]} beads; expected {no_beads}.')
        
        ids = np.arange(id0, id0 + no_beads)
        xyz[ids]   = P
        chain[ids] = c+1
        atype[ids[0]]   = 1      # chain end
        atype[ids[-1]] = 1       # chain end
        id0 += no_beads

    bond_i = np.zeros(Nbonds, dtype=int)
    bond_j = np.zeros(Nbonds, dtype=int)
    k0 = 0
    for c in range(no_chains):
        startId = c * no_beads
        ids_i = np.arange(startId, startId + no_beads - 1)
        ids_j = ids_i + 1
        nk = len(ids_i)
        bond_i[k0:k0+nk] = ids_i + 1
        bond_j[k0:k0+nk] = ids_j + 1
        k0 += nk

    # Write LAMMPS data file
    outfile = os.path.join(mainFolder, 'Input', f'topology_{fileTag}.txt')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    with open(outfile, 'w') as fid:
        fid.write(f'Topology SAW network: {no_chains} chains x {no_beads} beads = {Natom} atoms\n\n')
        fid.write(f'{Natom} atoms\n')
        fid.write(f'{Nbonds} bonds\n')
        fid.write('0 angles\n0 dihedrals\n0 impropers\n\n')
        fid.write('2 atom types\n1 bond types\n\n')
        fid.write(f'{xlo:.5e} {xhi:.5e} xlo xhi\n')
        fid.write(f'{ylo:.5e} {yhi:.5e} ylo yhi\n')
        fid.write(f'{zlo:.5e} {zhi:.5e} zlo zhi\n\n')
        fid.write('Masses\n\n1 1\n2 1\n\n')
        fid.write('Bond Coeffs\n\n1 %d 1 \n\n' % K)
        fid.write('Atoms\n\n')
        for i in range(Natom):
            fid.write(f'{i+1} {chain[i]} {atype[i]} {xyz[i,0]: .2f} {xyz[i,1]: .2f} {xyz[i,2]: .2f}\n')
        fid.write('\nVelocities\n\n')
        for i in range(Natom):
            fid.write(f'{i+1} 0 0 0\n')
        fid.write('\nBonds\n\n')
        for k in range(Nbonds):
            fid.write(f'{k+1} 1 {bond_i[k]} {bond_j[k]}\n')