import os
import numpy as np
import pickle

def generateSAWCordinates(no_chains, no_beads, phi, fileTag, mainFolder):
    coordFolder = os.path.join(mainFolder, 'Coordinates')
    if not os.path.isdir(coordFolder):
        os.makedirs(coordFolder)
    coordOut = os.path.join(coordFolder, f'coords_{fileTag}.pkl')

    tot_beads = no_chains * no_beads
    L_grid    = int(np.ceil((tot_beads / phi) ** (1 / 3)))
    print(f'[SAW] L_grid={L_grid}  tot_beads={tot_beads}  phi={phi:.3f}')

    maxStep, maxBacktracks, maxStartTrials, maxChainRestarts = 5, int(2e6), int(2e5), 200
    avoidEndBoundary = True

    offsetsByR = precomputeOffsets(maxStep, True)
    occ = np.zeros(L_grid**3, dtype=bool)
    chainPos = [None] * no_chains

    for c in range(no_chains):
        success = False
        for restart_attempt in range(maxChainRestarts):
            tempPos = np.zeros((no_beads, 3), dtype=np.int32)
            tempLin = np.zeros(no_beads, dtype=np.uint32)
            placedN = 0
            candPos = [None] * no_beads
            candLin = [None] * no_beads

            # start bead 
            startFound = False
            for _ in range(maxStartTrials):
                p0 = np.random.randint(1, L_grid + 1, size=3)
                lin0 = sub2ind3(L_grid, p0[0], p0[1], p0[2])
                if lin0 >= 0 and lin0 < L_grid**3 and not occ[lin0]:
                    if not avoidEndBoundary or not isBoundarySite(p0, L_grid):
                        startFound = True
                        break
            if not startFound:
                raise RuntimeError('No free start site (respecting end-boundary rule). Increase L_grid or reduce phi.')

            placedN = 1
            tempPos[0] = p0
            tempLin[0] = lin0
            occ[lin0] = True

            b = 1
            backtracks = 0
            while True:
                if b >= no_beads:
                    success = True
                    break
                prev = tempPos[b-1]
                if candPos[b-1] is None:
                    Pcan, Lcan = gatherCandidates(prev, L_grid, occ, offsetsByR)
                    candPos[b-1], candLin[b-1] = Pcan, Lcan
                
                Pcan = candPos[b-1]
                Lcan = candLin[b-1]
                
                if Pcan is not None and len(Pcan) > 0:
                    nxt = Pcan[-1]
                    lin = Lcan[-1]
                    candPos[b-1] = Pcan[:-1]
                    candLin[b-1] = Lcan[:-1]
                    if avoidEndBoundary and (b == no_beads-1) and isBoundarySite(nxt, L_grid):
                        continue
                    if not occ[lin]:
                        tempPos[b] = nxt
                        tempLin[b] = lin
                        occ[lin] = True
                        placedN = max(placedN, b+1)
                        candPos[b] = None
                        candLin[b] = None
                        b += 1
                        continue
                    else:
                        continue
                else:
                    backtracks += 1
                    if backtracks > maxBacktracks or b <= 1:
                        break
                    occ[int(tempLin[b-1])] = False
                    candPos[b-1] = None
                    candLin[b-1] = None
                    b -= 1
            if success:
                chainPos[c] = tempPos.copy()
                break
            else:
                for u in range(placedN):
                    occ[int(tempLin[u])] = False
        if not success:
            raise RuntimeError(f'Failed to build chain {c+1}. Try increasing L_grid or maxStep, or lowering phi.')

    combined = np.zeros((tot_beads, 5), dtype=np.int32)
    idx = 0
    for c in range(no_chains):
        P = chainPos[c]
        for b in range(no_beads):
            combined[idx, :] = [c+1, b+1, P[b,0], P[b,1], P[b,2]]
            idx += 1

    meta = {
        'no_chains': no_chains,
        'no_beads': no_beads,
        'tot_beads': tot_beads,
        'phi': phi,
        'L_grid': L_grid,
        'dim': 3,
        'pbc': True,
        'maxStep': maxStep,
        'fileTag': fileTag,
        'avoidEndBoundary': avoidEndBoundary
    }
    
    # Save as pickle (Python native format)
    data = {
        'chainPos': chainPos,
        'combined': combined,
        'meta': meta
    }
    with open(coordOut, 'wb') as f:
        pickle.dump(data, f)
    print(f'[SAW] Saved: {coordOut}')

def gatherCandidates(cur, L, occ, offsetsByR):
    """
    Gather valid candidates for next bead position.
    """
    Pall = []
    Lall = []
    for offs in offsetsByR:
        if len(offs) == 0:
            continue
        perm = np.random.permutation(offs.shape[0])
        offs_shuffled = offs[perm]
        for k in range(len(offs_shuffled)):
            nxt = cur + offs_shuffled[k]
            nxt = wrapPBC(nxt, L)
            lin = sub2ind3(L, nxt[0], nxt[1], nxt[2])
            
            if lin < 0 or lin >= L**3:
                continue
                
            if not occ[lin]:
                Pall.append(nxt)
                Lall.append(lin)
        if len(Pall) > 0:
            break
    if len(Pall) > 0:
        return (np.array(Pall, dtype=np.int32), np.array(Lall, dtype=np.uint32))
    else:
        return (np.array([], dtype=np.int32).reshape(0,3), np.array([], dtype=np.uint32))
    
def precomputeOffsets(maxStep, useChebyshevShell):
    """Precompute offset vectors for lattice neighbors."""
    offsetsByR = []
    for r in range(1, maxStep+1):
        vals = np.arange(-r, r+1)
        DX, DY, DZ = np.meshgrid(vals, vals, vals, indexing='ij')
        dx = DX.flatten()
        dy = DY.flatten()
        dz = DZ.flatten()
        nz = ~((dx==0) & (dy==0) & (dz==0))
        if useChebyshevShell:
            shell = np.max(np.abs(np.array([dx, dy, dz])), axis=0) == r
        else:
            shell = (np.abs(dx) + np.abs(dy) + np.abs(dz)) == r
        keep = nz & shell
        offsetsByR.append(np.stack([dx[keep], dy[keep], dz[keep]], axis=1).astype(np.int32))
    return offsetsByR

def isBoundarySite(p, L):
    """Check if position p is on the boundary."""
    return (p[0]==1 or p[0]==L or p[1]==1 or p[1]==L or p[2]==1 or p[2]==L)

def wrapPBC(nxt, L):
    """Apply periodic boundary conditions."""
    wrapped = np.zeros_like(nxt)
    for i in range(3):
        wrapped[i] = ((nxt[i] - 1) % L) + 1
    return wrapped

def sub2ind3(L, x, y, z):
    """Convert 1-based (x, y, z) to linear index."""
    x, y, z = int(x), int(y), int(z)
    lin = (x - 1) + (y - 1) * L + (z - 1) * L * L
    return lin